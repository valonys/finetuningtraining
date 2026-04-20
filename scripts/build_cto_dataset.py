#!/usr/bin/env python3
"""
scripts/build_cto_dataset.py
─────────────────────────────
Orchestrate all harvesters + Data Forge into a single CTO knowledge dataset.

Usage:
    python scripts/build_cto_dataset.py                      # defaults
    python scripts/build_cto_dataset.py --output ./data/processed/cto_v1.jsonl
    python scripts/build_cto_dataset.py --skip-arxiv --skip-youtube  # code + PDFs only

What it does:
    1. Harvest arXiv papers by keyword (abstract mode, configurable queries)
    2. Harvest Python code from local repos (AST + notebook extraction)
    3. Ingest PDFs already in data/uploads/ through the Data Forge pipeline
    4. Optionally harvest YouTube transcripts by keyword
    5. Optionally mix in an Alpaca subset for general capability
    6. Merge all JSONL files into a single training-ready dataset
    7. Split into train (90%) and test (10%) for eval

Output:
    data/processed/cto_knowledge_sft.jsonl        # training set
    data/processed/cto_knowledge_sft_test.jsonl   # held-out eval set
    data/processed/cto_knowledge_sft_preview.json # first 10 rows, pretty-printed
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_cto_dataset")

# ── Default arXiv queries (CTO-relevant domains) ──────────────────
DEFAULT_ARXIV_QUERIES = [
    "GRPO reinforcement learning language model",
    "LoRA low-rank adaptation fine-tuning",
    "retrieval augmented generation RAG",
    "LLM evaluation benchmark",
    "knowledge distillation language model",
    "agentic LLM tool use planning",
]

# ── Default YouTube queries ───────────────────────────────────────
DEFAULT_YOUTUBE_QUERIES = [
    "LLM fine-tuning tutorial 2024",
    "RAG retrieval augmented generation explained",
    "GRPO reinforcement learning from verifiable rewards",
    "LoRA QLoRA efficient fine-tuning",
    "vLLM inference optimization",
]

# ── Default code repos (Manning books) ────────────────────────────
DEFAULT_CODE_PATHS = [
    "./data/manning/reasoning-from-scratch-main",
    "./data/manning/ai-evaluations-main",
    "./data/manning/deep-learning-in-motion-master",
]


def harvest_arxiv(queries: list[str], max_per_query: int = 15, output_dir: str = "./data/uploads") -> int:
    """Harvest arXiv abstracts and return total rows written."""
    from app.harvesters.arxiv import ArxivHarvester
    h = ArxivHarvester()
    total = 0
    for q in queries:
        try:
            report = h.harvest(q, max_results=max_per_query, mode="abstract", output_dir=output_dir)
            total += len(report.harvested)
            logger.info(f"  arXiv '{q}': {len(report.harvested)} harvested, {len(report.skipped)} skipped")
        except Exception as e:
            logger.warning(f"  arXiv '{q}' failed: {e}")
    return total


def harvest_code(paths: list[str], output_dir: str = "./data/uploads") -> int:
    """Harvest code from directories and return total units extracted."""
    from app.harvesters.code import CodeHarvester
    h = CodeHarvester()
    total = 0
    for p in paths:
        if not Path(p).is_dir():
            logger.warning(f"  Code path not found (skip): {p}")
            continue
        try:
            report = h.harvest_directory(
                p, strategy="all", source_label=Path(p).name, output_dir=output_dir,
            )
            total += report.total_units
            logger.info(f"  Code '{p}': {report.total_units} units from {report.files_scanned} files")
        except Exception as e:
            logger.warning(f"  Code '{p}' failed: {e}")
    return total


def harvest_youtube(queries: list[str], max_per_query: int = 5, output_dir: str = "./data/uploads") -> int:
    """Harvest YouTube transcripts and return total rows written."""
    from app.harvesters.youtube import YouTubeHarvester
    h = YouTubeHarvester()
    total = 0
    for q in queries:
        try:
            report = h.harvest(q, max_results=max_per_query, output_dir=output_dir)
            total += len(report.harvested)
            logger.info(f"  YouTube '{q}': {len(report.harvested)} harvested, {len(report.skipped)} skipped")
        except Exception as e:
            logger.warning(f"  YouTube '{q}' failed: {e}")
    return total


def build_forge_dataset(
    upload_dir: str = "./data/uploads",
    output_dir: str = "./data/processed",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    system_prompt: str = "You are a senior AI engineer and CTO specialising in LLM post-training, evaluation, and production inference.",
    target_size: int | None = None,
) -> str | None:
    """Run the Data Forge build_dataset on everything in upload_dir.
    Returns the output JSONL path, or None on failure."""
    from app.data_forge.dataset_builder import DatasetBuilder

    uploads = list(Path(upload_dir).glob("*"))
    txt_files = [str(f) for f in uploads if f.suffix in (".txt", ".pdf", ".docx", ".md", ".html")]
    if not txt_files:
        logger.warning("  No ingestible files found in %s", upload_dir)
        return None

    logger.info(f"  Forge: building dataset from {len(txt_files)} files")
    builder = DatasetBuilder(
        paths=txt_files,
        task="sft",
        base_model=base_model,
        system_prompt=system_prompt,
        synth_qa=True,
        filter_noise=True,
        target_size=target_size,
        output_dir=output_dir,
    )
    try:
        result = builder.build()
        logger.info(f"  Forge: {result.get('num_examples', 0)} examples -> {result.get('output_path', '?')}")
        return result.get("output_path")
    except Exception as e:
        logger.exception(f"  Forge build failed: {e}")
        return None


def merge_jsonl_files(paths: list[str], output_path: str) -> int:
    """Merge multiple JSONL files into one, deduplicating by instruction text.
    Returns total row count."""
    seen_instructions: set = set()
    rows: list[dict] = []

    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Deduplicate by instruction text (case-insensitive)
                key = (row.get("instruction") or row.get("text") or "").strip().lower()
                if key and key not in seen_instructions:
                    seen_instructions.add(key)
                    rows.append(row)

    # Shuffle for training diversity
    random.shuffle(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"  Merged: {len(rows)} rows -> {output_path}")
    return len(rows)


def split_train_test(input_path: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split a JSONL file into train and test sets. Returns (train_path, test_path)."""
    rows = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                rows.append(line)

    random.shuffle(rows)
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    train_path = input_path  # overwrite the original with the train split
    test_path = input_path.replace(".jsonl", "_test.jsonl")

    with open(train_path, "w") as f:
        f.writelines(train_rows)
    with open(test_path, "w") as f:
        f.writelines(test_rows)

    logger.info(f"  Split: {len(train_rows)} train, {len(test_rows)} test")
    return train_path, test_path


def write_preview(jsonl_path: str, n: int = 10):
    """Write a pretty-printed JSON preview of the first N rows."""
    preview_path = jsonl_path.replace(".jsonl", "_preview.json")
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line.strip()))
    with open(preview_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info(f"  Preview: {preview_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Build the CTO knowledge SFT dataset")
    parser.add_argument("--output", default="./data/processed/cto_knowledge_sft.jsonl")
    parser.add_argument("--upload-dir", default="./data/uploads")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target-size", type=int, default=None, help="Target dataset size for Q/A synth")
    parser.add_argument("--skip-arxiv", action="store_true")
    parser.add_argument("--skip-youtube", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-forge", action="store_true")
    parser.add_argument("--code-paths", nargs="*", default=None, help="Override code repo paths")
    parser.add_argument("--arxiv-queries", nargs="*", default=None)
    parser.add_argument("--youtube-queries", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    partial_outputs: list[str] = []

    logger.info("=" * 60)
    logger.info("CTO Knowledge Dataset Builder")
    logger.info("=" * 60)

    # ── 1. arXiv ──────────────────────────────────────────────
    if not args.skip_arxiv:
        logger.info("\n[1/4] Harvesting arXiv papers...")
        queries = args.arxiv_queries or DEFAULT_ARXIV_QUERIES
        harvest_arxiv(queries, output_dir=args.upload_dir)

    # ── 2. Code ───────────────────────────────────────────────
    if not args.skip_code:
        logger.info("\n[2/4] Harvesting code from repos...")
        code_paths = args.code_paths or DEFAULT_CODE_PATHS
        harvest_code(code_paths, output_dir=args.upload_dir)
        # Code harvester writes its own .jsonl — find it
        for f in Path(args.upload_dir).glob("code_harvest_*.jsonl"):
            partial_outputs.append(str(f))

    # ── 3. YouTube ────────────────────────────────────────────
    if not args.skip_youtube:
        logger.info("\n[3/4] Harvesting YouTube transcripts...")
        queries = args.youtube_queries or DEFAULT_YOUTUBE_QUERIES
        harvest_youtube(queries, output_dir=args.upload_dir)

    # ── 4. Forge (PDFs + text files) ──────────────────────────
    if not args.skip_forge:
        logger.info("\n[4/4] Running Data Forge on uploaded files...")
        forge_output = build_forge_dataset(
            upload_dir=args.upload_dir,
            output_dir=str(Path(args.output).parent),
            base_model=args.base_model,
            target_size=args.target_size,
        )
        if forge_output:
            partial_outputs.append(forge_output)

    # ── 5. Merge ──────────────────────────────────────────────
    logger.info("\n[Merge] Combining all sources...")
    total = merge_jsonl_files(partial_outputs, args.output)

    if total == 0:
        logger.warning("No rows produced. Check that source data exists.")
        return

    # ── 6. Split ──────────────────────────────────────────────
    logger.info("\n[Split] Creating train/test split (90/10)...")
    train_path, test_path = split_train_test(args.output)

    # ── 7. Preview ────────────────────────────────────────────
    write_preview(train_path)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Test:  {test_path}")
    logger.info(f"  Total: {total} rows")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
