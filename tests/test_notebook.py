"""Notebook remains portable JSON with list-form source cells."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _notebook() -> dict:
    return json.loads((ROOT / "notebooks/kaggle_run.ipynb").read_text(encoding="utf-8"))


def test_notebook() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert isinstance(cell["source"], list)
            assert all(isinstance(line, str) for line in cell["source"])


def test_notebook_references_existing_files() -> None:
    code = "".join(
        "".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"
    )
    referenced = set(re.findall(r"(?:scripts|configs)/[\w.-]+\.(?:py|json)", code))
    assert referenced, "notebook should reference scripts or configs"
    missing = sorted(path for path in referenced if not (ROOT / path).exists())
    assert not missing
