"""
app/harvesters/code.py
──────────────────────
Directory scanner for .py and .ipynb files -> SFT-ready instruction/response
pairs written as .jsonl.

Design:

  Python files are parsed with the `ast` module to extract functions and
  classes together with their signatures, docstrings, decorators, and full
  source text.  Notebook files are parsed with `json` to pair consecutive
  markdown + code cells into natural instruction/response pairs.

  Each extracted unit is formatted according to a *strategy* (implement,
  explain, review, docstring, or all) and written as one JSONL row so the
  Data Forge can pick it up via the standard ingest flow.

  Syntax errors in source files are caught and logged -- they never crash the
  harvester.  The same goes for malformed notebooks.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class CodeUnit:
    """A single extracted code artefact (function, class, or notebook cell)."""
    file_path: str
    name: str                     # function/class name or "notebook_cell_N"
    code: str                     # full source text
    context: str                  # docstring or markdown cell text
    unit_type: str                # "function" | "class" | "notebook_cell" | "module"
    source_label: str = ""


@dataclass
class CodeHarvestReport:
    """Summary returned by CodeHarvester.harvest_directory()."""
    units: list[CodeUnit] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    total_units: int = 0
    output_path: str = ""


# ── Strategy formatters ────────────────────────────────────────────

_STRATEGIES = ("implement", "explain", "review", "docstring")


def _format_implement(unit: CodeUnit) -> dict:
    description = unit.context.strip() if unit.context.strip() else "as described below"
    if unit.unit_type == "class":
        instruction = (
            f"Implement a Python class called `{unit.name}` that {description}."
        )
    elif unit.unit_type == "notebook_cell":
        # For notebooks the markdown IS the natural instruction.
        instruction = unit.context.strip()
    else:
        instruction = (
            f"Implement a Python function called `{unit.name}` that {description}."
        )
    return {"instruction": instruction, "response": unit.code}


def _format_explain(unit: CodeUnit) -> dict:
    instruction = (
        f"Explain what the following Python code does:\n"
        f"```python\n{unit.code}\n```"
    )
    explanation_parts: list[str] = []
    if unit.context.strip():
        explanation_parts.append(unit.context.strip())
    kind = "class" if unit.unit_type == "class" else "function"
    if unit.unit_type == "notebook_cell":
        kind = "code block"
    explanation_parts.append(
        f"This {kind} `{unit.name}` processes the logic shown above."
    )
    response = " ".join(explanation_parts)
    return {"instruction": instruction, "response": response}


def _format_review(unit: CodeUnit) -> dict:
    instruction = (
        f"Review the following Python code for correctness, style, and "
        f"potential improvements:\n```python\n{unit.code}\n```"
    )
    response_parts = []
    if unit.context.strip():
        response_parts.append(f"Summary: {unit.context.strip()}")
    response_parts.append(
        f"The {unit.unit_type} `{unit.name}` appears to be correctly "
        f"structured.  Consider adding type hints and docstrings where "
        f"missing, and ensure edge cases are handled."
    )
    return {"instruction": instruction, "response": " ".join(response_parts)}


def _format_docstring(unit: CodeUnit) -> dict:
    instruction = (
        f"Write a comprehensive docstring for the following Python code:\n"
        f"```python\n{unit.code}\n```"
    )
    if unit.context.strip():
        response = unit.context.strip()
    else:
        response = (
            f'"""{unit.name}: performs the operations defined in its body."""'
        )
    return {"instruction": instruction, "response": response}


_STRATEGY_FN = {
    "implement": _format_implement,
    "explain": _format_explain,
    "review": _format_review,
    "docstring": _format_docstring,
}


def _apply_strategies(unit: CodeUnit, strategy: str) -> list[dict]:
    """Return one row per applicable strategy."""
    rows: list[dict] = []
    targets = list(_STRATEGIES) if strategy == "all" else [strategy]
    for strat in targets:
        # Notebooks always use implement (markdown is the natural instruction)
        if unit.unit_type == "notebook_cell" and strat != "implement":
            continue
        fn = _STRATEGY_FN.get(strat)
        if fn is None:
            continue
        pair = fn(unit)
        pair["source"] = unit.file_path
        pair["type"] = unit.unit_type
        pair["strategy"] = strat
        if unit.source_label:
            pair["source_label"] = unit.source_label
        rows.append(pair)
    return rows


# ── AST helpers ────────────────────────────────────────────────────

def _source_lines(source: str, node: ast.AST) -> str:
    """Extract the source text for an AST node using line numbers."""
    lines = source.splitlines(keepends=True)
    # ast line numbers are 1-indexed
    start = node.lineno - 1
    end = getattr(node, "end_lineno", None)
    if end is not None:
        segment = "".join(lines[start:end])
    else:
        segment = "".join(lines[start:])
    return textwrap.dedent(segment)


def _decorator_text(source: str, node: ast.AST) -> str:
    """Return the decorator lines that precede a function/class node."""
    decorators = getattr(node, "decorator_list", [])
    if not decorators:
        return ""
    lines = source.splitlines(keepends=True)
    first_dec_line = decorators[0].lineno - 1
    node_line = node.lineno - 1
    return "".join(lines[first_dec_line:node_line])


def _is_boilerplate_init(node: ast.FunctionDef) -> bool:
    """
    Return True if this __init__ body is nothing but `self.x = x` assignments
    (with an optional docstring as the first statement).
    """
    if node.name != "__init__":
        return False
    body = list(node.body)
    # Skip leading docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Constant, ast.Str)):
        body = body[1:]
    if not body:
        return True  # empty init
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            return False
        # Check each target is self.xxx
        for target in stmt.targets:
            if not (isinstance(target, ast.Attribute) and
                    isinstance(target.value, ast.Name) and
                    target.value.id == "self"):
                return False
        # Check the value is a simple Name (i.e. self.x = x, not self.x = x + 1)
        if not isinstance(stmt.value, ast.Name):
            return False
    return True


def _body_line_count(node: ast.AST) -> int:
    """Approximate line count of the body of a function/class."""
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end - node.lineno + 1
    return 0


# ── .py extraction ─────────────────────────────────────────────────

def _extract_py_units(
    file_path: str,
    source: str,
    *,
    min_lines: int,
    max_lines: int,
    source_label: str,
) -> list[CodeUnit]:
    """Parse a .py file and return CodeUnits for each function/class."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        logger.warning("Syntax error in %s: %s", file_path, exc)
        return []

    units: list[CodeUnit] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        name = node.name
        body_lines = _body_line_count(node)

        # Skip short bodies
        if body_lines < min_lines:
            continue

        # Skip boilerplate __init__
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_boilerplate_init(node):
                continue

        docstring = ast.get_docstring(node) or ""

        # Full source text (decorators + definition)
        dec_text = _decorator_text(source, node)
        body_text = _source_lines(source, node)

        if body_lines > max_lines:
            # Include a truncation marker but still emit the unit header
            code_text = dec_text + body_text
            truncation_note = (
                f"\n# ... [TRUNCATED: {body_lines} lines exceeds "
                f"max_lines={max_lines}] ...\n"
            )
            # Keep only up to max_lines of the body
            code_lines = code_text.splitlines(keepends=True)
            code_text = "".join(code_lines[:max_lines]) + truncation_note
        else:
            code_text = dec_text + body_text

        unit_type = "class" if isinstance(node, ast.ClassDef) else "function"

        units.append(CodeUnit(
            file_path=file_path,
            name=name,
            code=code_text,
            context=docstring,
            unit_type=unit_type,
            source_label=source_label,
        ))

    return units


# ── .ipynb extraction ──────────────────────────────────────────────

_IMPORT_RE = re.compile(
    r"^\s*(import\s|from\s\S+\s+import\s|%pip\s|!pip\s|%conda\s|!conda\s)",
)


def _is_only_imports_or_installs(code: str) -> bool:
    """Return True if every non-blank line is an import or pip install."""
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not lines:
        return True
    return all(_IMPORT_RE.match(ln) for ln in lines)


def _extract_ipynb_units(
    file_path: str,
    *,
    min_lines: int,
    source_label: str,
) -> list[CodeUnit]:
    """Parse a .ipynb notebook and pair markdown cells with following code cells."""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            nb = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load notebook %s: %s", file_path, exc)
        return []

    cells = nb.get("cells", [])
    units: list[CodeUnit] = []
    cell_counter = 0

    i = 0
    while i < len(cells):
        cell = cells[i]
        cell_type = cell.get("cell_type", "")

        if cell_type == "markdown":
            md_source = "".join(cell.get("source", []))

            # Skip markdown cells shorter than 20 chars (just headers)
            if len(md_source.strip()) < 20:
                i += 1
                continue

            # Look ahead for a following code cell
            if i + 1 < len(cells) and cells[i + 1].get("cell_type") == "code":
                code_cell = cells[i + 1]
                code_source = "".join(code_cell.get("source", []))
                code_line_count = len([
                    ln for ln in code_source.splitlines() if ln.strip()
                ])

                # Skip code cells shorter than min_lines
                if code_line_count < min_lines:
                    i += 2
                    continue

                # Skip code cells that are only imports or pip installs
                if _is_only_imports_or_installs(code_source):
                    i += 2
                    continue

                cell_counter += 1
                units.append(CodeUnit(
                    file_path=file_path,
                    name=f"notebook_cell_{cell_counter}",
                    code=code_source,
                    context=md_source.strip(),
                    unit_type="notebook_cell",
                    source_label=source_label,
                ))
                i += 2  # consumed both cells
                continue

        i += 1

    return units


# ── Harvester ──────────────────────────────────────────────────────

class CodeHarvester:
    """
    Scans directories for .py and .ipynb files and extracts SFT-ready
    instruction/response pairs.

    Typical use:

        h = CodeHarvester()
        report = h.harvest_directory(
            "./my_project",
            strategy="implement",
            output_dir="./data/uploads",
        )
        print(report.total_units, "units ->", report.output_path)

    The output .jsonl file lands in `output_dir` so the Data Forge can
    ingest it via the standard flow.
    """

    def harvest_directory(
        self,
        path: str,
        *,
        extensions: list[str] = None,
        strategy: str = "implement",
        min_lines: int = 5,
        max_lines: int = 200,
        source_label: str = "",
        output_dir: str = "./data/uploads",
    ) -> CodeHarvestReport:
        """
        Walk `path` recursively, extract code units, write .jsonl output.

        Parameters
        ----------
        path : str
            Root directory to scan.
        extensions : list[str]
            File extensions to include (default ``[".py", ".ipynb"]``).
        strategy : str
            One of "implement", "explain", "review", "docstring", "all".
        min_lines : int
            Minimum non-blank lines for a unit to be included.
        max_lines : int
            Maximum lines before a unit is truncated.
        source_label : str
            Optional label added to every row (e.g. project name).
        output_dir : str
            Directory for the output .jsonl file.
        """
        if extensions is None:
            extensions = [".py", ".ipynb"]

        valid_strategies = {"implement", "explain", "review", "docstring", "all"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {sorted(valid_strategies)}."
            )

        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        report = CodeHarvestReport()
        all_units: list[CodeUnit] = []
        files_scanned = 0
        files_skipped = 0

        # Walk directory
        for dirpath, _dirnames, filenames in os.walk(root):
            # Skip hidden directories and __pycache__
            base = os.path.basename(dirpath)
            if base.startswith(".") or base == "__pycache__":
                continue
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in extensions:
                    continue

                fpath = os.path.join(dirpath, fname)
                label = source_label or str(root.name)

                if ext == ".py":
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                            source = fh.read()
                    except OSError as exc:
                        logger.warning("Cannot read %s: %s", fpath, exc)
                        files_skipped += 1
                        continue

                    units = _extract_py_units(
                        fpath, source,
                        min_lines=min_lines,
                        max_lines=max_lines,
                        source_label=label,
                    )
                    if units:
                        all_units.extend(units)
                        files_scanned += 1
                    else:
                        files_scanned += 1  # scanned but no units extracted

                elif ext == ".ipynb":
                    units = _extract_ipynb_units(
                        fpath,
                        min_lines=min_lines,
                        source_label=label,
                    )
                    if units:
                        all_units.extend(units)
                        files_scanned += 1
                    else:
                        files_scanned += 1

                else:
                    files_skipped += 1

        # ── Build JSONL rows ─────────────────────────────────────
        rows: list[dict] = []
        for unit in all_units:
            rows.extend(_apply_strategies(unit, strategy))

        # ── Write output ─────────────────────────────────────────
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic filename from the scanned directory name
        dir_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", root.name)[:60]
        out_path = out_dir / f"code_harvest_{dir_slug}.jsonl"

        with open(out_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            "Code harvest complete: %d units from %d files -> %s",
            len(all_units), files_scanned, out_path,
        )

        report.units = all_units
        report.files_scanned = files_scanned
        report.files_skipped = files_skipped
        report.total_units = len(all_units)
        report.output_path = str(out_path)
        return report
