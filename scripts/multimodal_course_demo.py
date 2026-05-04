#!/usr/bin/env python3
"""Index the course transcript files into the multimodal pipeline.

Usage:
    python scripts/multimodal_course_demo.py \
        --source-dir ~/Documents/"Building Multimodal Data Pipelines" \
        --collection multimodal-course
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.multimodal import MultimodalPipeline, PipelineConfig
from app.multimodal.processors import TextFileProcessor
from app.multimodal.schemas import Modality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--collection", default="multimodal-course")
    parser.add_argument("--tenant-id", default="public")
    parser.add_argument("--query", default="How does multimodal RAG work across audio slides and video?")
    args = parser.parse_args()

    root = Path(args.source_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"source dir not found: {root}")

    pipeline = MultimodalPipeline(
        config=PipelineConfig(
            tenant_id=args.tenant_id,
            collection=args.collection,
            chunk_target_chars=1200,
            chunk_overlap_chars=160,
        )
    )
    processor = TextFileProcessor()

    records = []
    for path in sorted(root.glob("*.txt")):
        records.extend(
            processor.process(
                str(path),
                tenant_id=args.tenant_id,
                collection=args.collection,
                source_type=_source_type_for(path),
                title=path.name,
            )
        )

    chunks = pipeline.index_records(records)
    print(f"Indexed {len(records)} records / {len(chunks)} chunks into {args.collection}")
    print("Stats:", pipeline.vector_store.stats(tenant_id=args.tenant_id, collection=args.collection))
    print(f"\nQuery: {args.query}")
    for idx, result in enumerate(pipeline.search(args.query, top_k=5), start=1):
        src = result.chunk.source
        preview = result.chunk.text[:220].replace("\n", " ")
        print(f"{idx}. {src.source_type.value} {src.title} score={result.score:.4f}")
        print(f"   {preview}")


def _source_type_for(path: Path) -> Modality:
    # Course transcript mapping:
    # 2/3 cover the general + ASR/OCR foundation, 4/5 video/VLM,
    # 6 multimodal RAG. They are all text files, but we preserve the
    # dominant lesson modality so filtered retrieval can be demonstrated.
    if path.name in {"4.txt", "5.txt"}:
        return Modality.VIDEO
    if path.name == "6.txt":
        return Modality.TEXT
    return Modality.AUDIO


if __name__ == "__main__":
    main()
