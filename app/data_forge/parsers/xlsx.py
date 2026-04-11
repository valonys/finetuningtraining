"""XLSX / XLS / CSV / TSV parsers."""
from __future__ import annotations

from ..ingest import IngestedRecord


def parse_xlsx(path: str, **_kw) -> IngestedRecord:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas not installed") from e

    sheets = pd.read_excel(path, sheet_name=None, dtype=str).fillna("")
    parts: list[str] = []
    tables: list[list[list[str]]] = []

    for name, df in sheets.items():
        parts.append(f"## Sheet: {name}")
        parts.append(df.to_csv(index=False, sep="|").strip())
        rows = [list(df.columns)] + df.astype(str).values.tolist()
        tables.append(rows)

    return IngestedRecord(
        source=path,
        source_type="xlsx",
        text="\n\n".join(parts),
        tables=tables,
        metadata={"num_sheets": len(sheets), "sheets": list(sheets.keys())},
    )


def parse_csv(path: str, **_kw) -> IngestedRecord:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas not installed") from e

    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    rows = [list(df.columns)] + df.astype(str).values.tolist()

    return IngestedRecord(
        source=path,
        source_type="csv",
        text=df.to_csv(index=False, sep="|").strip(),
        tables=[rows],
        metadata={"num_rows": len(df), "columns": list(df.columns)},
    )
