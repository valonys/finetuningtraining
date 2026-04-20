"""
Unit tests for the Code harvester.

All tests use temporary directories with synthetic .py and .ipynb files.
We verify:
  * AST extraction of functions and classes from .py files
  * Notebook markdown+code cell pairing from .ipynb files
  * min_lines filtering (short functions are skipped)
  * Boilerplate __init__ methods are skipped
  * JSONL output format is correct
  * Empty directories produce an empty report without crashing
  * Syntax errors in .py files are caught (no crash)
  * Strategy formatting produces correct instruction/response pairs
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from app.harvesters.code import (
    CodeHarvester,
    CodeHarvestReport,
    CodeUnit,
    _extract_ipynb_units,
    _extract_py_units,
    _is_boilerplate_init,
    _is_only_imports_or_installs,
)


# ── Helpers ────────────────────────────────────────────────────────

def _write_py(tmp_path: Path, name: str, content: str) -> Path:
    """Write a .py file into tmp_path and return its path."""
    fpath = tmp_path / name
    fpath.write_text(textwrap.dedent(content), encoding="utf-8")
    return fpath


def _write_ipynb(tmp_path: Path, name: str, cells: list[dict]) -> Path:
    """Write a minimal .ipynb file and return its path."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells,
    }
    fpath = tmp_path / name
    fpath.write_text(json.dumps(nb), encoding="utf-8")
    return fpath


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": []}


# ── .py extraction tests ──────────────────────────────────────────

class TestPyExtraction:

    def test_extract_function(self, tmp_path):
        src = '''\
def compute_area(width, height):
    """Calculate the area of a rectangle."""
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    area = width * height
    return area
'''
        fpath = _write_py(tmp_path, "geometry.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="test",
        )
        assert len(units) == 1
        u = units[0]
        assert u.name == "compute_area"
        assert u.unit_type == "function"
        assert u.context == "Calculate the area of a rectangle."
        assert "def compute_area" in u.code
        assert u.source_label == "test"

    def test_extract_class(self, tmp_path):
        src = '''\
class DataProcessor:
    """Processes raw data into structured output."""
    def __init__(self, config):
        self.config = config

    def process(self, data):
        """Run the processing pipeline."""
        validated = self._validate(data)
        transformed = self._transform(validated)
        return transformed

    def _validate(self, data):
        if not data:
            raise ValueError("Empty data")
        return data

    def _transform(self, data):
        return [item.strip() for item in data]
'''
        fpath = _write_py(tmp_path, "processor.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="",
        )
        # Should find the class and the methods that meet min_lines
        class_units = [u for u in units if u.unit_type == "class"]
        func_units = [u for u in units if u.unit_type == "function"]
        assert len(class_units) == 1
        assert class_units[0].name == "DataProcessor"
        assert class_units[0].context == "Processes raw data into structured output."
        # process method has 3+ lines
        assert any(u.name == "process" for u in func_units)

    def test_skip_short_function(self, tmp_path):
        src = '''\
def tiny():
    return 1

def bigger(x):
    """Does something bigger."""
    a = x + 1
    b = a * 2
    c = b - 3
    d = c + 4
    return d
'''
        fpath = _write_py(tmp_path, "mixed.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=5, max_lines=200, source_label="",
        )
        names = [u.name for u in units]
        assert "tiny" not in names
        assert "bigger" in names

    def test_skip_boilerplate_init(self, tmp_path):
        src = '''\
class Simple:
    """Simple class."""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def do_work(self):
        """Does the actual work."""
        result = self.a + self.b
        result *= self.c
        result += 1
        result -= 2
        return result
'''
        fpath = _write_py(tmp_path, "simple.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="",
        )
        names = [u.name for u in units]
        assert "__init__" not in names
        assert "do_work" in names

    def test_init_with_logic_is_kept(self, tmp_path):
        src = '''\
class Complex:
    def __init__(self, path):
        self.path = path
        self.data = self._load(path)
        if not self.data:
            raise ValueError("No data found")
        self.processed = False
        self.cache = {}
'''
        fpath = _write_py(tmp_path, "complex.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="",
        )
        names = [u.name for u in units]
        # This __init__ has a method call and an if statement, not boilerplate
        assert "__init__" in names

    def test_syntax_error_does_not_crash(self, tmp_path):
        bad_src = "def broken(\n    x y z\n"
        fpath = _write_py(tmp_path, "broken.py", bad_src)
        units = _extract_py_units(
            str(fpath), bad_src, min_lines=1, max_lines=200, source_label="",
        )
        assert units == []

    def test_async_function_extracted(self, tmp_path):
        src = '''\
import asyncio

async def fetch_data(url):
    """Fetch data from a URL asynchronously."""
    response = await asyncio.sleep(0)
    data = {"url": url}
    result = process(data)
    return result

def process(data):
    return data
'''
        fpath = _write_py(tmp_path, "async_mod.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="",
        )
        names = [u.name for u in units]
        assert "fetch_data" in names

    def test_max_lines_truncation(self, tmp_path):
        # Generate a function with many lines
        body_lines = "\n".join(f"    x_{i} = {i}" for i in range(50))
        src = f"def big_function():\n    '''Big function.'''\n{body_lines}\n    return x_0\n"
        fpath = _write_py(tmp_path, "big.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=10, source_label="",
        )
        assert len(units) == 1
        assert "TRUNCATED" in units[0].code

    def test_decorated_function(self, tmp_path):
        src = '''\
def my_decorator(func):
    """A simple decorator."""
    def wrapper(*args, **kwargs):
        print("before")
        result = func(*args, **kwargs)
        print("after")
        return result
    return wrapper

@my_decorator
def greet(name):
    """Greet someone by name."""
    message = f"Hello, {name}!"
    print(message)
    return message
'''
        fpath = _write_py(tmp_path, "deco.py", src)
        units = _extract_py_units(
            str(fpath), src, min_lines=3, max_lines=200, source_label="",
        )
        greet_units = [u for u in units if u.name == "greet"]
        assert len(greet_units) == 1
        assert "@my_decorator" in greet_units[0].code


# ── .ipynb extraction tests ───────────────────────────────────────

class TestIpynbExtraction:

    def test_markdown_code_pairing(self, tmp_path):
        cells = [
            _md_cell("## Load and preprocess the dataset for training"),
            _code_cell(
                "import pandas as pd\n\n"
                "df = pd.read_csv('data.csv')\n"
                "df = df.dropna()\n"
                "df['target'] = df['target'].astype(int)\n"
                "print(df.shape)\n"
            ),
        ]
        fpath = _write_ipynb(tmp_path, "demo.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=3, source_label="nb")
        assert len(units) == 1
        u = units[0]
        assert u.unit_type == "notebook_cell"
        assert u.name == "notebook_cell_1"
        assert "Load and preprocess" in u.context
        assert "pd.read_csv" in u.code
        assert u.source_label == "nb"

    def test_short_markdown_skipped(self, tmp_path):
        cells = [
            _md_cell("# Title"),  # too short (< 20 chars)
            _code_cell("x = 1\ny = 2\nz = 3\nw = 4\nv = 5\n"),
        ]
        fpath = _write_ipynb(tmp_path, "short_md.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=3, source_label="")
        assert len(units) == 0

    def test_short_code_skipped(self, tmp_path):
        cells = [
            _md_cell("This is a detailed markdown explanation of what follows"),
            _code_cell("x = 1\n"),  # only 1 line, below min_lines=3
        ]
        fpath = _write_ipynb(tmp_path, "short_code.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=3, source_label="")
        assert len(units) == 0

    def test_imports_only_skipped(self, tmp_path):
        cells = [
            _md_cell("Install and import required dependencies for the project"),
            _code_cell(
                "import numpy as np\n"
                "import pandas as pd\n"
                "from sklearn.model_selection import train_test_split\n"
                "import matplotlib.pyplot as plt\n"
                "from pathlib import Path\n"
            ),
        ]
        fpath = _write_ipynb(tmp_path, "imports.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=1, source_label="")
        assert len(units) == 0

    def test_pip_installs_skipped(self, tmp_path):
        cells = [
            _md_cell("Install the packages we will need for this notebook"),
            _code_cell(
                "!pip install transformers\n"
                "!pip install datasets\n"
                "%pip install accelerate\n"
            ),
        ]
        fpath = _write_ipynb(tmp_path, "installs.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=1, source_label="")
        assert len(units) == 0

    def test_multiple_pairs(self, tmp_path):
        cells = [
            _md_cell("First, create the training configuration object"),
            _code_cell(
                "config = {\n"
                "    'lr': 1e-4,\n"
                "    'epochs': 3,\n"
                "    'batch_size': 16,\n"
                "    'warmup': 100,\n"
                "}\n"
            ),
            _md_cell("Next, build the data loader with shuffling enabled"),
            _code_cell(
                "loader = DataLoader(\n"
                "    dataset,\n"
                "    batch_size=config['batch_size'],\n"
                "    shuffle=True,\n"
                "    num_workers=4,\n"
                ")\n"
            ),
        ]
        fpath = _write_ipynb(tmp_path, "multi.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=3, source_label="")
        assert len(units) == 2
        assert units[0].name == "notebook_cell_1"
        assert units[1].name == "notebook_cell_2"

    def test_malformed_notebook_no_crash(self, tmp_path):
        fpath = tmp_path / "bad.ipynb"
        fpath.write_text("{invalid json!!", encoding="utf-8")
        units = _extract_ipynb_units(str(fpath), min_lines=1, source_label="")
        assert units == []

    def test_consecutive_markdown_cells(self, tmp_path):
        """When two markdown cells appear in a row, only the one before a code cell pairs."""
        cells = [
            _md_cell("This is the first markdown explanation block"),
            _md_cell("This is the second markdown explanation block, the real one"),
            _code_cell(
                "result = compute()\n"
                "print(result)\n"
                "validate(result)\n"
                "store(result)\n"
                "log(result)\n"
            ),
        ]
        fpath = _write_ipynb(tmp_path, "consec.ipynb", cells)
        units = _extract_ipynb_units(str(fpath), min_lines=3, source_label="")
        assert len(units) == 1
        # The second markdown cell pairs with the code cell
        assert "second markdown" in units[0].context


# ── Harvester integration tests ───────────────────────────────────

class TestCodeHarvester:

    def test_empty_directory(self, tmp_path):
        h = CodeHarvester()
        report = h.harvest_directory(str(tmp_path), output_dir=str(tmp_path / "out"))
        assert isinstance(report, CodeHarvestReport)
        assert report.total_units == 0
        assert report.files_scanned == 0
        assert report.units == []
        # Output file should still be created (empty .jsonl)
        assert Path(report.output_path).exists()

    def test_harvest_py_files(self, tmp_path):
        src = '''\
def analyze(data, threshold=0.5):
    """Analyze data against a threshold."""
    results = []
    for item in data:
        if item > threshold:
            results.append(item)
    return results
'''
        _write_py(tmp_path, "analysis.py", src)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="implement",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        assert report.total_units >= 1
        assert report.files_scanned >= 1
        assert Path(report.output_path).exists()

        # Verify JSONL output
        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert len(rows) >= 1
        row = rows[0]
        assert "instruction" in row
        assert "response" in row
        assert row["strategy"] == "implement"
        assert row["type"] == "function"
        assert "analyze" in row["instruction"]
        assert "def analyze" in row["response"]

    def test_harvest_ipynb_files(self, tmp_path):
        cells = [
            _md_cell("Create a function that computes the Fibonacci sequence up to n terms"),
            _code_cell(
                "def fibonacci(n):\n"
                "    seq = [0, 1]\n"
                "    while len(seq) < n:\n"
                "        seq.append(seq[-1] + seq[-2])\n"
                "    return seq[:n]\n"
                "\n"
                "print(fibonacci(10))\n"
            ),
        ]
        _write_ipynb(tmp_path, "fib.ipynb", cells)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="implement",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        assert report.total_units >= 1

        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

        nb_rows = [r for r in rows if r["type"] == "notebook_cell"]
        assert len(nb_rows) >= 1
        assert "Fibonacci" in nb_rows[0]["instruction"]
        assert "def fibonacci" in nb_rows[0]["response"]

    def test_strategy_explain(self, tmp_path):
        src = '''\
def bubble_sort(arr):
    """Sort an array using the bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''
        _write_py(tmp_path, "sort.py", src)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="explain",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert len(rows) >= 1
        row = rows[0]
        assert row["strategy"] == "explain"
        assert "Explain what the following" in row["instruction"]
        assert "bubble sort" in row["response"].lower()

    def test_strategy_all(self, tmp_path):
        src = '''\
def add(a, b):
    """Add two numbers and return the sum."""
    result = a + b
    validated = result is not None
    return result if validated else 0
'''
        _write_py(tmp_path, "math_ops.py", src)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="all",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        # strategy="all" should produce one row per strategy for each unit
        strategies_present = {r["strategy"] for r in rows}
        assert "implement" in strategies_present
        assert "explain" in strategies_present
        assert "review" in strategies_present
        assert "docstring" in strategies_present

    def test_strategy_all_notebook_only_implement(self, tmp_path):
        """Notebook cells should only get 'implement' strategy even with strategy='all'."""
        cells = [
            _md_cell("Compute the mean and standard deviation of a dataset column"),
            _code_cell(
                "import statistics\n"
                "data = [1, 2, 3, 4, 5]\n"
                "mean = statistics.mean(data)\n"
                "stdev = statistics.stdev(data)\n"
                "print(f'mean={mean}, stdev={stdev}')\n"
            ),
        ]
        _write_ipynb(tmp_path, "stats.ipynb", cells)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="all",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        nb_rows = [r for r in rows if r["type"] == "notebook_cell"]
        # All notebook rows should be "implement" only
        assert all(r["strategy"] == "implement" for r in nb_rows)

    def test_jsonl_format_complete(self, tmp_path):
        """Every JSONL row must have all required keys."""
        src = '''\
def transform(data, factor=2):
    """Transform data by multiplying each element by a factor."""
    output = []
    for item in data:
        output.append(item * factor)
    return output
'''
        _write_py(tmp_path, "transform.py", src)
        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="implement",
            source_label="myproject",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        with open(report.output_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert len(rows) >= 1
        required_keys = {"instruction", "response", "source", "type", "strategy"}
        for row in rows:
            assert required_keys.issubset(row.keys()), (
                f"Missing keys: {required_keys - row.keys()}"
            )
            assert row["source_label"] == "myproject"

    def test_invalid_strategy_raises(self, tmp_path):
        h = CodeHarvester()
        with pytest.raises(ValueError, match="Unknown strategy"):
            h.harvest_directory(str(tmp_path), strategy="bogus")

    def test_nonexistent_directory_raises(self):
        h = CodeHarvester()
        with pytest.raises(FileNotFoundError):
            h.harvest_directory("/nonexistent/path/xyz_does_not_exist")

    def test_mixed_py_and_ipynb(self, tmp_path):
        """Harvest a directory containing both .py and .ipynb files."""
        py_src = '''\
def cleanup(text):
    """Remove leading and trailing whitespace from text."""
    stripped = text.strip()
    normalized = " ".join(stripped.split())
    return normalized
'''
        _write_py(tmp_path, "utils.py", py_src)

        cells = [
            _md_cell("Demonstrate how to use the cleanup function on sample text"),
            _code_cell(
                "text = '  hello   world  '\n"
                "result = cleanup(text)\n"
                "assert result == 'hello world'\n"
                "print(f'Cleaned: {result}')\n"
                "print('Done')\n"
            ),
        ]
        _write_ipynb(tmp_path, "demo.ipynb", cells)

        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            strategy="implement",
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        types = {u.unit_type for u in report.units}
        assert "function" in types
        assert "notebook_cell" in types
        assert report.files_scanned >= 2

    def test_pycache_skipped(self, tmp_path):
        """Files inside __pycache__ directories are not scanned."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        src = "def cached_fn():\n    return 1\n    x = 2\n    y = 3\n    z = 4\n"
        (cache_dir / "mod.py").write_text(src, encoding="utf-8")

        h = CodeHarvester()
        report = h.harvest_directory(
            str(tmp_path),
            min_lines=3,
            output_dir=str(tmp_path / "out"),
        )
        assert report.total_units == 0


# ── Utility function tests ────────────────────────────────────────

class TestUtilities:

    def test_is_only_imports(self):
        assert _is_only_imports_or_installs("import os\nimport sys\n")
        assert _is_only_imports_or_installs("from pathlib import Path\nimport json\n")
        assert _is_only_imports_or_installs("!pip install torch\n%pip install numpy\n")
        assert not _is_only_imports_or_installs("import os\nx = os.getcwd()\n")
        assert not _is_only_imports_or_installs("result = compute()\n")
        assert _is_only_imports_or_installs("")  # empty is trivially all-imports
        assert _is_only_imports_or_installs("   \n\n  \n")  # blank lines only

    def test_is_boilerplate_init(self):
        import ast as _ast

        # Boilerplate init
        src = textwrap.dedent('''\
            class A:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
        ''')
        tree = _ast.parse(src)
        cls = tree.body[0]
        init = cls.body[0]
        assert _is_boilerplate_init(init) is True

        # Non-boilerplate init (has a method call)
        src2 = textwrap.dedent('''\
            class B:
                def __init__(self, path):
                    self.data = self.load(path)
        ''')
        tree2 = _ast.parse(src2)
        cls2 = tree2.body[0]
        init2 = cls2.body[0]
        assert _is_boilerplate_init(init2) is False

        # Non-init method should always return False
        src3 = textwrap.dedent('''\
            class C:
                def process(self, x):
                    self.x = x
        ''')
        tree3 = _ast.parse(src3)
        cls3 = tree3.body[0]
        method = cls3.body[0]
        assert _is_boilerplate_init(method) is False
