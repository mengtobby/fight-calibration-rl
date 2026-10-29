"""Notebook remains portable JSON with list-form source cells."""

import json
from pathlib import Path


def test_notebook() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/kaggle_run.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert isinstance(cell["source"], list)
            assert all(isinstance(line, str) for line in cell["source"])
    for script in ("prepare_data", "run_baselines", "train_grpo", "train_sft", "predict", "report"):
        assert (root / f"scripts/{script}.py").exists()
