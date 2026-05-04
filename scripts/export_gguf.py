#!/usr/bin/env python3
"""
scripts/export_gguf.py
──────────────────────
Operator CLI for A1 GGUF export. Merges a LoRA adapter into its base
model and produces a quantized GGUF + metadata sidecar + rollback pointer.

Example:
    python scripts/export_gguf.py \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter outputs/ai_llm \\
        --output-dir outputs/ai_llm/artifacts \\
        --quant Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="HF id of the base model the adapter was trained on")
    parser.add_argument("--adapter", required=True, help="Path to the trained adapter folder")
    parser.add_argument("--output-dir", required=True, help="Where to write the GGUF + sidecar + latest.gguf")
    parser.add_argument("--quant", default="Q4_K_M", help="llama-quantize scheme (default: Q4_K_M)")
    parser.add_argument("--llama-cpp-path", default=None, help="Override vendored llama.cpp path")
    parser.add_argument("--artifact-name", default=None, help="Override artifact stem (default: adapter folder name)")
    parser.add_argument("--quiet", action="store_true", help="Only print the final JSON result")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from app.trainers.export import merge_and_export
    result = merge_and_export(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        quant=args.quant,
        llama_cpp_path=args.llama_cpp_path,
        artifact_name=args.artifact_name,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
