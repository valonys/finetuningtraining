#!/usr/bin/env python3
"""
scripts/batch_pipeline.py
─────────────────────────
Operator CLI for the A2 batch pipeline (collect → forge → train → eval → deploy).

Examples:
    # Fresh end-to-end run for a domain
    python scripts/batch_pipeline.py --domain ai_llm

    # Skip collect+forge, just retrain & deploy
    python scripts/batch_pipeline.py --domain ai_llm --stages train,eval,deploy

    # Resume a crashed run from where it left off
    python scripts/batch_pipeline.py --resume 20260503-093000-ai_llm
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


def _load_domain_config(domain: str) -> dict:
    """Load configs/domains/<domain>.yaml if it exists, else empty dict."""
    cfg_path = REPO_ROOT / "configs" / "domains" / f"{domain}.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as exc:
        raise SystemExit(f"failed to load {cfg_path}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--domain", help="Domain name (loads configs/domains/<domain>.yaml)")
    parser.add_argument("--resume", help="Resume an existing run by run_id")
    parser.add_argument(
        "--stages",
        default="collect,forge,train,eval,deploy",
        help="Comma-separated stage names to run (default: all)",
    )
    parser.add_argument("--runs-root", default="outputs/runs", help="Where run dirs live")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.domain and not args.resume:
        parser.error("either --domain or --resume is required")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from app.pipeline import PipelineRunner
    from app.pipeline.stages import select_stages

    stage_names = [s.strip() for s in args.stages.split(",") if s.strip()]
    stages = select_stages(stage_names)

    if args.resume:
        # Domain inferred from manifest on resume.
        runner = PipelineRunner(domain="<from-manifest>", runs_root=args.runs_root)
        ctx = runner.resume_run(args.resume)
        # Rebuild runner with the correct domain + config from manifest.
        runner = PipelineRunner(
            domain=ctx.domain,
            config=ctx.config,
            runs_root=args.runs_root,
        )
    else:
        cfg = _load_domain_config(args.domain)
        runner = PipelineRunner(
            domain=args.domain,
            config=cfg,
            runs_root=args.runs_root,
        )
        ctx = runner.start_run(stages_requested=stage_names)

    report = runner.execute(stages, ctx)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
