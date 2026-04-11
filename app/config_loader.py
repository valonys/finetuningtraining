"""
app/config_loader.py
────────────────────
Domain-config CRUD for ValonyLabs Studio.

A "domain" is any post-training engagement: `asset_integrity`, `customer_grasps`,
`ai_llm`, `legal_nda_review`, `medical_intake`, etc. Each domain is a single
YAML file in `configs/domains/` and produces an adapter at `outputs/<name>/`.

Users create domains via:
  * `create_domain_config(...)`  — this module (Python)
  * `scripts/new_domain.py`       — CLI
  * `POST /v1/domains/configs`    — REST API
  * Gradio Studio's "🏷️ Domains" tab

The `configs/domains/` folder ships empty. Seed examples live in
`configs/domains/examples/` and are **never auto-loaded** — they only show
up when the user explicitly asks to seed from one.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
CONFIG_ROOT = os.environ.get("VALONY_CONFIG_ROOT", "configs/domains")
EXAMPLES_DIR = os.path.join(CONFIG_ROOT, "examples")
TEMPLATE_FILE = os.path.join(CONFIG_ROOT, "_template.yaml")

# Reserved filenames that must not be treated as user configs.
_RESERVED = {"_template.yaml", "_template.yml"}

# Valid domain name: lowercase, starts with letter, alnum + underscore, ≤64.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────
class DomainConfigError(ValueError):
    """Raised for invalid names, duplicate configs, missing files, etc."""


# ──────────────────────────────────────────────────────────────
# Read / list
# ──────────────────────────────────────────────────────────────
def load_domain_config(name: str, *, include_examples: bool = False) -> Dict[str, Any]:
    """
    Load a domain config by name.

    Resolution order:
      1. `configs/domains/<name>.yaml`  (the user's own config — strict default)
      2. `configs/domains/examples/<name>.yaml`  (only when `include_examples=True`)

    The `include_examples` path is useful for demo notebooks that want to work
    out-of-the-box, but production code should keep it `False` so that a
    missing config fails loudly instead of silently loading a seed example.

    Args:
        name: domain name, or a full/absolute path to a YAML file.
        include_examples: if True, fall back to `examples/<name>.yaml`.
    """
    # Allow a full path for advanced use
    if os.path.isabs(name) or os.sep in name:
        path = name
    else:
        fname = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
        path = os.path.join(CONFIG_ROOT, fname)
        if not os.path.exists(path) and include_examples:
            alt = os.path.join(EXAMPLES_DIR, fname)
            if os.path.exists(alt):
                path = alt

    if not os.path.exists(path):
        hint = _not_found_hint(name)
        raise FileNotFoundError(
            f"Domain config '{name}' not found at {path}.{hint}"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "domain_name" not in cfg:
        cfg["domain_name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def list_domain_configs() -> List[str]:
    """Return the names of user-created configs in `configs/domains/`.

    Excludes the template file and anything in the `examples/` subfolder.
    """
    if not os.path.isdir(CONFIG_ROOT):
        return []
    names: list[str] = []
    for entry in sorted(os.listdir(CONFIG_ROOT)):
        if entry in _RESERVED:
            continue
        full = os.path.join(CONFIG_ROOT, entry)
        if os.path.isdir(full):
            continue
        if not entry.endswith((".yaml", ".yml")):
            continue
        names.append(os.path.splitext(entry)[0])
    return names


def list_domain_examples() -> List[str]:
    """Return the names of seed examples in `configs/domains/examples/`."""
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    names: list[str] = []
    for entry in sorted(os.listdir(EXAMPLES_DIR)):
        if entry.endswith((".yaml", ".yml")):
            names.append(os.path.splitext(entry)[0])
    return names


def domain_config_exists(name: str) -> bool:
    """Does `configs/domains/<name>.yaml` exist as a user config?"""
    _validate_name(name)
    return os.path.exists(_path_for(name))


def get_domain_template() -> Dict[str, Any]:
    """Return the template blueprint as a dict (for UI forms, API, etc.)."""
    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(f"Template missing at {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────
# Create
# ──────────────────────────────────────────────────────────────
def create_domain_config(
    name: str,
    *,
    system_prompt: Optional[str] = None,
    constitution: Optional[List[str]] = None,
    training_args: Optional[Dict[str, Any]] = None,
    dpo_args: Optional[Dict[str, Any]] = None,
    orpo_args: Optional[Dict[str, Any]] = None,
    kto_args: Optional[Dict[str, Any]] = None,
    grpo_args: Optional[Dict[str, Any]] = None,
    copy_from: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Create a new domain config at `configs/domains/<name>.yaml`.

    Args:
        name: domain name. Must match `^[a-z][a-z0-9_]{0,63}$`.
        system_prompt: the persona string. If omitted and `copy_from` is set,
            the example's system_prompt is used verbatim.
        constitution: list of guardrail rules. Same fallback as above.
        training_args / dpo_args / orpo_args / kto_args / grpo_args:
            hyperparameter overrides. If omitted, the seed's values are kept.
        copy_from: name of an example (in `configs/domains/examples/`) or an
            existing user config to seed from. If None, the `_template.yaml`
            blueprint is used.
        overwrite: if True, replace an existing config. Default False — raises
            `DomainConfigError` on collision.

    Returns:
        Absolute path to the created YAML.
    """
    _validate_name(name)

    target = _path_for(name)
    if os.path.exists(target) and not overwrite:
        raise DomainConfigError(
            f"Domain config '{name}' already exists at {target}. "
            f"Pass overwrite=True to replace it."
        )

    # Start from the seed (example or template)
    if copy_from:
        seed = _load_seed(copy_from)
    else:
        try:
            seed = get_domain_template()
        except FileNotFoundError:
            seed = _default_template_dict()

    cfg: Dict[str, Any] = {
        "domain_name": name,
        "system_prompt": system_prompt if system_prompt is not None else seed.get("system_prompt", ""),
        "constitution": constitution if constitution is not None else list(seed.get("constitution") or []),
        "training_args": _merge(seed.get("training_args"), training_args),
        "dpo_args":      _merge(seed.get("dpo_args"),      dpo_args),
        "orpo_args":     _merge(seed.get("orpo_args"),     orpo_args),
        "kto_args":      _merge(seed.get("kto_args"),      kto_args),
        "grpo_args":     _merge(seed.get("grpo_args"),     grpo_args),
    }

    os.makedirs(CONFIG_ROOT, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return target


def copy_example_to_domain(example_name: str, new_name: Optional[str] = None, *, overwrite: bool = False) -> str:
    """Convenience: copy an example file verbatim to `configs/domains/<new_name>.yaml`.

    Useful when the user wants the full commented-out fields of the example
    and plans to edit the YAML directly in their editor.
    """
    src = os.path.join(EXAMPLES_DIR, f"{example_name}.yaml")
    if not os.path.exists(src):
        raise DomainConfigError(f"Example '{example_name}' not found at {src}")
    target_name = new_name or example_name
    _validate_name(target_name)
    dst = _path_for(target_name)
    if os.path.exists(dst) and not overwrite:
        raise DomainConfigError(f"Domain config '{target_name}' already exists at {dst}")
    os.makedirs(CONFIG_ROOT, exist_ok=True)
    shutil.copyfile(src, dst)
    # Rewrite domain_name inside the copy so it matches the new filename
    _rewrite_domain_name(dst, target_name)
    return dst


# ──────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────
def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise DomainConfigError(
            f"Invalid domain name '{name}'. "
            f"Must start with a lowercase letter and contain only "
            f"[a-z0-9_], max 64 characters."
        )


def _path_for(name: str) -> str:
    return os.path.join(CONFIG_ROOT, f"{name}.yaml")


def _load_seed(copy_from: str) -> Dict[str, Any]:
    """Try examples/ first, then the user's own configs folder."""
    ex = os.path.join(EXAMPLES_DIR, f"{copy_from}.yaml")
    if os.path.exists(ex):
        with open(ex, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    own = _path_for(copy_from)
    if os.path.exists(own):
        with open(own, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise DomainConfigError(
        f"Seed '{copy_from}' not found in examples/ or in the user configs folder. "
        f"Available examples: {', '.join(list_domain_examples()) or '(none)'}. "
        f"Available user configs: {', '.join(list_domain_configs()) or '(none)'}."
    )


def _merge(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    if override:
        out.update(override)
    return out


def _rewrite_domain_name(path: str, new_name: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["domain_name"] = new_name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _not_found_hint(name: str) -> str:
    user = list_domain_configs()
    examples = list_domain_examples()
    parts = []
    if user:
        parts.append(f"Available configs: {', '.join(user)}.")
    else:
        parts.append("No user configs yet — create one with "
                     "`python scripts/new_domain.py create <name>` "
                     "or `POST /v1/domains/configs`.")
    if examples:
        parts.append(f"Seed examples: {', '.join(examples)}.")
    return " " + " ".join(parts)


def _default_template_dict() -> Dict[str, Any]:
    """Last-resort fallback if `_template.yaml` has been removed."""
    return {
        "domain_name": "your_domain_name_here",
        "system_prompt": "You are a helpful assistant.",
        "constitution": [],
        "training_args": {
            "lora_r": 16, "lora_alpha": 32, "learning_rate": 2.0e-4,
            "batch_size": 2, "gradient_accumulation_steps": 4,
            "num_train_epochs": 1, "max_steps": -1, "max_seq_length": 2048,
        },
        "dpo_args": {
            "learning_rate": 5.0e-5, "beta": 0.1, "num_train_epochs": 1,
            "batch_size": 1, "gradient_accumulation_steps": 8, "max_length": 2048,
        },
        "orpo_args": {
            "learning_rate": 8.0e-6, "beta": 0.1, "num_train_epochs": 1,
            "batch_size": 1, "gradient_accumulation_steps": 8, "max_length": 2048,
        },
        "kto_args": {
            "learning_rate": 5.0e-6, "beta": 0.1,
            "desirable_weight": 1.0, "undesirable_weight": 1.0,
            "num_train_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 8,
        },
        "grpo_args": {
            "num_generations": 8, "temperature": 0.8, "max_new_tokens": 400,
            "learning_rate": 5.0e-6, "num_train_epochs": 2,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
        },
    }
