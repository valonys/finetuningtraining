"""
app/trainers/hub.py
───────────────────
Push trained LoRA adapters to the HuggingFace Hub.

Usage from a notebook (after `result = trainer.train()`):

    from app.trainers.hub import push_adapter_to_hub
    url = push_adapter_to_hub(
        adapter_dir=result['adapter_path'],
        repo_id='amiguel/qwen-0.5b-ai_llm-sft',
        private=True,
    )

Reads HF_TOKEN from the env (the Colab bootstrap already populates it
from Colab Secrets). Auto-creates the repo when missing, and writes a
README with the run's metadata so the model card isn't empty.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def push_adapter_to_hub(
    adapter_dir: str,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Create-or-update an HF model repo and upload the adapter folder.

    Args:
        adapter_dir: Path returned by `trainer.train()` as
            `result['adapter_path']` (e.g. `outputs/ai_llm`). Should
            contain `adapter_config.json`, `adapter_model.safetensors`,
            and the tokenizer files.
        repo_id: Target on the Hub. Format: `<owner>/<name>`. The owner
            must be either your username or an org you can write to.
        private: True (default) to keep the repo unlisted.
        commit_message: Optional commit message; auto-generated if None.
        token: HF token. Defaults to `HF_TOKEN` env var (set by the
            Colab bootstrap from your Colab Secrets).
        metadata: Optional dict written to a model card section. The
            notebooks pass the trainer's `result` here so the card
            shows method/backend/template/loss/etc.

    Returns:
        The full https URL of the uploaded repo.
    """
    folder = Path(adapter_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tok = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not tok:
        raise RuntimeError(
            "No HF token found. Either:\n"
            "  * pass token='hf_xxx' explicitly, OR\n"
            "  * set HF_TOKEN env var (Colab Secrets pickup runs in the\n"
            "    notebook bootstrap cell -- restart runtime if you just\n"
            "    added the Secret), OR\n"
            "  * `huggingface-cli login` from the shell."
        )

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. Run `pip install huggingface_hub`."
        ) from e

    # ── Create-or-touch the repo ──────────────────────────────────
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=tok,
    )
    logger.info(f"🤗 Repo ready: {repo_id} (private={private})")

    # ── Stage upload directory with a generated README ───────────
    # We don't write into `folder` directly so we don't pollute the
    # caller's outputs/ tree with a model card.
    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        for item in folder.iterdir():
            dst = staging_path / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        readme_path = staging_path / "README.md"
        if not readme_path.exists():
            readme_path.write_text(_render_model_card(repo_id, metadata or {}))

        commit = commit_message or _default_commit(metadata or {})
        upload_folder(
            folder_path=str(staging_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit,
            token=tok,
        )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"🤗 Adapter pushed: {url}")
    return url


def _default_commit(meta: Dict[str, Any]) -> str:
    method = (meta.get("method") or "sft").upper()
    samples = meta.get("samples")
    loss = meta.get("final_loss")
    bits = []
    if samples is not None:
        bits.append(f"{samples} samples")
    if loss is not None:
        try:
            bits.append(f"loss={float(loss):.3f}")
        except Exception:
            pass
    detail = ", ".join(bits) if bits else "trained adapter"
    return f"Upload {method} adapter ({detail}) — ValonyLabs Studio"


def _render_model_card(repo_id: str, meta: Dict[str, Any]) -> str:
    """A minimal model card — base model, training method, run metrics."""
    base_model = meta.get("base_model") or meta.get("base_model_id") or "unknown"
    method = (meta.get("method") or "sft").lower()
    backend = meta.get("backend", "trl")
    template = meta.get("template", "auto")
    samples = meta.get("samples", "?")
    loss = meta.get("final_loss")
    loss_str = f"{float(loss):.4f}" if loss is not None else "n/a"

    # YAML front-matter is what makes the model render with the proper
    # "base model" link and tag chips on the Hub.
    yaml = (
        "---\n"
        f"base_model: {base_model}\n"
        "library_name: peft\n"
        "tags:\n"
        f"  - {method}\n"
        "  - lora\n"
        "  - valonylabs-studio\n"
        "---\n"
    )
    body = (
        f"# {repo_id.split('/')[-1]}\n\n"
        f"LoRA adapter trained with **ValonyLabs Studio** "
        f"(method=`{method}`, backend=`{backend}`, template=`{template}`).\n\n"
        f"## Training summary\n\n"
        f"| field | value |\n"
        f"|---|---|\n"
        f"| Base model | `{base_model}` |\n"
        f"| Method | `{method}` |\n"
        f"| Samples | {samples} |\n"
        f"| Final training loss | {loss_str} |\n"
        f"| Backend | {backend} |\n"
        f"| Template | {template} |\n\n"
        f"## How to load\n\n"
        f"```python\n"
        f"from peft import PeftModel\n"
        f"from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f"base = AutoModelForCausalLM.from_pretrained('{base_model}', device_map='auto')\n"
        f"model = PeftModel.from_pretrained(base, '{repo_id}')\n"
        f"tok   = AutoTokenizer.from_pretrained('{repo_id}')\n"
        f"```\n"
    )
    return yaml + "\n" + body
