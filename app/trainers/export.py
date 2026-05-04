"""
app/trainers/export.py
──────────────────────
Merge a LoRA adapter into its base model and export to GGUF for runtime use
by llama.cpp / Ollama. Closes A1 of the Lane A blueprint
(see ``docs/SPRINTS.md``).

Pipeline:
  1. Load base model + tokenizer (CPU; merge does not need a GPU).
  2. Attach the PEFT adapter and call ``merge_and_unload()`` to bake it
     into the base weights.
  3. Save the merged model as a full HF checkpoint to a temp dir.
  4. Run llama.cpp's ``convert_hf_to_gguf.py`` to produce an f16 GGUF.
  5. Invoke ``llama-quantize`` to compress (default ``Q4_K_M``).
  6. Compute SHA-256, write a sidecar ``<artifact>.metadata.json``.
  7. Update the ``latest.gguf`` rollback pointer in ``output_dir``.

The llama.cpp toolchain is vendored separately by
``scripts/install_llamacpp.sh`` which clones a pinned commit into
``~/.local/llama.cpp`` (override with ``VALONY_LLAMA_CPP_PATH``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def merge_and_export(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    *,
    quant: str = "Q4_K_M",
    llama_cpp_path: str | None = None,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    """Merge a LoRA adapter and export the result to a quantized GGUF.

    Args:
        base_model_id: HF id of the base model the adapter was trained on.
        adapter_path: Folder containing ``adapter_config.json`` and weights
            (typically the ``result['adapter_path']`` from a trainer run).
        output_dir: Where to write the GGUF + metadata + ``latest.gguf``
            symlink. Created if missing.
        quant: llama.cpp quant scheme (``Q4_K_M``, ``Q5_K_M``, ``Q8_0``,
            ``F16``, ...). Passed through to ``llama-quantize``.
        llama_cpp_path: Override path to the vendored llama.cpp checkout.
            Falls back to ``VALONY_LLAMA_CPP_PATH`` env var, then to
            ``~/.local/llama.cpp``.
        artifact_name: Stem for the output files. Defaults to the adapter
            folder name.

    Returns:
        ``{'gguf_path', 'metadata_path', 'sha256', 'latest_pointer',
           'quant', 'base_model_id', 'adapter_sha256', 'exported_at'}``.

    Raises:
        FileNotFoundError: adapter dir missing or llama.cpp toolchain
            not installed.
        subprocess.CalledProcessError: convert or quantize step failed.
    """
    adapter = Path(adapter_path).expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter dir invalid: {adapter} (no adapter_config.json). "
            f"Did you pass the trainer's result['adapter_path']?"
        )

    llama_dir = _resolve_llama_cpp_path(llama_cpp_path)
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = _find_quantize_binary(llama_dir)
    if not convert_script.is_file():
        raise FileNotFoundError(
            f"llama.cpp converter missing: {convert_script}\n"
            f"Run scripts/install_llamacpp.sh to install."
        )
    if quantize_bin is None:
        raise FileNotFoundError(
            f"llama-quantize binary missing under {llama_dir}/build/bin/. "
            f"Run scripts/install_llamacpp.sh to build it."
        )

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_name or adapter.name
    quant_lower = quant.lower()
    final_gguf = out_dir / f"{stem}-{quant_lower}.gguf"

    adapter_hash = _hash_dir(adapter)

    with tempfile.TemporaryDirectory(prefix="valony-export-") as tmp:
        merged_dir = Path(tmp) / "merged"
        f16_path = Path(tmp) / f"{stem}-f16.gguf"

        # 1-3 — load, merge, save merged HF checkpoint
        _merge_lora_into_base(
            base_model_id=base_model_id,
            adapter_path=adapter,
            merged_dir=merged_dir,
        )

        # 4 — HF -> f16 GGUF
        logger.info("🔧 Converting merged model to f16 GGUF")
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # 5 — quantize (skip if user asked for raw f16)
        if quant.upper() in {"F16", "FP16"}:
            shutil.move(str(f16_path), str(final_gguf))
        else:
            logger.info(f"🔧 Quantizing -> {quant}")
            subprocess.run(
                [str(quantize_bin), str(f16_path), str(final_gguf), quant],
                check=True,
                capture_output=True,
                text=True,
            )

    # 6 — checksum + metadata sidecar
    file_hash = _hash_file(final_gguf)
    metadata = {
        "base_model_id": base_model_id,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_hash,
        "quant": quant,
        "gguf_filename": final_gguf.name,
        "file_sha256": file_hash,
        "file_bytes": final_gguf.stat().st_size,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llama_cpp_path": str(llama_dir),
    }
    metadata_path = out_dir / f"{stem}-{quant_lower}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # 7 — rollback pointer
    latest = _update_latest_pointer(out_dir, final_gguf)

    logger.info(f"✅ Exported {final_gguf.name} ({metadata['file_bytes']:,} bytes)")
    return {
        "gguf_path": str(final_gguf),
        "metadata_path": str(metadata_path),
        "sha256": file_hash,
        "latest_pointer": str(latest),
        "quant": quant,
        "base_model_id": base_model_id,
        "adapter_sha256": adapter_hash,
        "exported_at": metadata["exported_at"],
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _resolve_llama_cpp_path(explicit: str | None) -> Path:
    raw = explicit or os.environ.get("VALONY_LLAMA_CPP_PATH") or "~/.local/llama.cpp"
    return Path(raw).expanduser().resolve()


def _find_quantize_binary(llama_dir: Path) -> Path | None:
    """llama.cpp's quantize binary moved around; probe the known paths."""
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "quantize",
        llama_dir / "llama-quantize",
        llama_dir / "quantize",
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _merge_lora_into_base(
    *, base_model_id: str, adapter_path: Path, merged_dir: Path
) -> None:
    """Load base + adapter on CPU, merge, save full HF checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"📥 Loading base model: {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    logger.info(f"📥 Attaching adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))

    logger.info("🔀 Merging adapter weights into base")
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hash_dir(path: Path) -> str:
    """Stable hash over file contents in a directory (sorted by relative path)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        h.update(p.relative_to(path).as_posix().encode())
        h.update(b"\0")
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def _update_latest_pointer(out_dir: Path, target: Path) -> Path:
    """Maintain ``latest.gguf`` -> most recent export. Symlink on POSIX,
    plain copy on Windows where symlinks need elevated privileges."""
    latest = out_dir / "latest.gguf"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(target.name)  # relative target keeps things portable
    except (OSError, NotImplementedError):
        shutil.copy2(target, latest)
    return latest
