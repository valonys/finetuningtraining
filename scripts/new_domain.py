#!/usr/bin/env python3
"""
scripts/new_domain.py
─────────────────────
CLI for creating and inspecting domain configs.

Subcommands
───────────
    create NAME           Create a new config at configs/domains/<name>.yaml
    list                  List user configs + seed examples
    show NAME             Dump one config to stdout (YAML)
    show-template         Dump the blueprint template to stdout (YAML)
    show-example NAME     Dump one example (asset_integrity, customer_grasps, ai_llm, ...)

Examples
────────
    # Blank domain with a short system prompt
    python scripts/new_domain.py create asset_integrity \\
        --system "You are an Asset Integrity engineer for offshore assets." \\
        --rule "Always prioritise safety over uptime." \\
        --rule "Cite API 570 and ASME standards when relevant."

    # Seed from a shipped example, then override the system prompt
    python scripts/new_domain.py create my_legal \\
        --copy-from asset_integrity \\
        --system "You are a contracts attorney..."

    # Copy verbatim (keeps full commented-out fields for editing by hand)
    python scripts/new_domain.py create my_support --copy-from customer_grasps --verbatim

    # List what you have
    python scripts/new_domain.py list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# Let the script run from anywhere inside the repo
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import yaml  # noqa: E402

from app.config_loader import (  # noqa: E402
    CONFIG_ROOT,
    EXAMPLES_DIR,
    DomainConfigError,
    copy_example_to_domain,
    create_domain_config,
    get_domain_template,
    list_domain_configs,
    list_domain_examples,
    load_domain_config,
)


# ──────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────
def cmd_create(args: argparse.Namespace) -> int:
    constitution = list(args.rule) if args.rule else None

    try:
        if args.verbatim:
            if not args.copy_from:
                _fail("--verbatim requires --copy-from")
            path = copy_example_to_domain(
                args.copy_from, new_name=args.name, overwrite=args.overwrite
            )
        else:
            path = create_domain_config(
                name=args.name,
                system_prompt=args.system,
                constitution=constitution,
                copy_from=args.copy_from,
                overwrite=args.overwrite,
            )
    except DomainConfigError as e:
        _fail(str(e))

    print(f"✅ Created domain config: {path}")
    print()
    print("Preview:")
    with open(path, "r", encoding="utf-8") as f:
        print(textwrap.indent(f.read(), "    "))
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    configs = list_domain_configs()
    examples = list_domain_examples()
    print(f"📂 User configs ({CONFIG_ROOT}):")
    if configs:
        for c in configs:
            print(f"   - {c}  →  {CONFIG_ROOT}/{c}.yaml")
    else:
        print("   (none yet — create one with `create NAME`)")
    print()
    print(f"🌱 Seed examples ({EXAMPLES_DIR}):")
    if examples:
        for e in examples:
            print(f"   - {e}")
    else:
        print("   (none shipped)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        cfg = load_domain_config(args.name)
    except FileNotFoundError as e:
        _fail(str(e))
    _print_yaml(cfg)
    return 0


def cmd_show_template(_args: argparse.Namespace) -> int:
    try:
        tpl = get_domain_template()
    except FileNotFoundError as e:
        _fail(str(e))
    _print_yaml(tpl)
    return 0


def cmd_show_example(args: argparse.Namespace) -> int:
    path = os.path.join(EXAMPLES_DIR, f"{args.name}.yaml")
    if not os.path.exists(path):
        examples = list_domain_examples()
        _fail(f"Example '{args.name}' not found. Available: {', '.join(examples) or '(none)'}")
    with open(path, "r", encoding="utf-8") as f:
        sys.stdout.write(f.read())
    return 0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _print_yaml(obj) -> None:
    sys.stdout.write(yaml.safe_dump(obj, sort_keys=False, default_flow_style=False, allow_unicode=True))


def _fail(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="new_domain.py",
        description="Manage ValonyLabs Studio domain configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="Create a new domain config YAML")
    c.add_argument("name", help="Domain name (e.g., asset_integrity, ai_llm)")
    c.add_argument("--system", "-s", default=None,
                   help="System prompt / persona")
    c.add_argument("--rule", "-r", action="append", default=None,
                   help="Constitution rule (repeatable)")
    c.add_argument("--copy-from", "-c", default=None,
                   help="Seed from an example (asset_integrity, customer_grasps, ai_llm, ...)")
    c.add_argument("--verbatim", action="store_true",
                   help="Copy the --copy-from example byte-for-byte (keeps comments). "
                        "Only --name is respected; other flags are ignored.")
    c.add_argument("--overwrite", action="store_true",
                   help="Replace an existing config")
    c.set_defaults(func=cmd_create)

    l = sub.add_parser("list", help="List user configs and seed examples")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="Dump one user config (YAML)")
    s.add_argument("name")
    s.set_defaults(func=cmd_show)

    t = sub.add_parser("show-template", help="Dump the blueprint template (YAML)")
    t.set_defaults(func=cmd_show_template)

    e = sub.add_parser("show-example", help="Dump one example (verbatim YAML)")
    e.add_argument("name")
    e.set_defaults(func=cmd_show_example)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
