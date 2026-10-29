"""Training configuration contracts without model dependencies."""

import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fightcal.train.config import GRPORunConfig, SFTRunConfig, load_config


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.json")))
def test_shipped_configs(path):
    cfg = load_config(path)
    assert type(cfg).from_dict(cfg.to_dict()) == cfg
    assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 2


def base():
    return dict(arm="test", train_path="train", val_path="val", output_dir="out")


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": 1},
        {"seed": -1},
        {"seed": True},
        {"model_name": ""},
        {"anonymize": 1},
        {"learning_rate": float("nan")},
        {"max_steps": 0},
        {"max_train_bouts": -1},
        {"lora_dropout": 1},
        {"beta": -1},
        {"reward": "other"},
        {"num_generations": 3},
        {"kind": "sft"},
        {"temperature": 0},
        {"lora_r": 1.5},
    ],
)
def test_invalid_grpo(changes):
    with pytest.raises(ValueError):
        GRPORunConfig.from_dict(base() | changes)


def test_bad_loaders(tmp_path):
    for data in ([], {}, {"kind": "other"}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_config(path)
    for data in ([], {}):
        with pytest.raises(ValueError):
            GRPORunConfig.from_dict(data)
    for changes in ({"kind": "grpo"}, {"num_epochs": 0}, {"max_seq_length": 0}):
        with pytest.raises(ValueError):
            SFTRunConfig.from_dict(base() | changes)


def test_lazy_training_imports():
    for name in ("grpo", "sft"):
        assert importlib.import_module(f"fightcal.train.{name}")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import fightcal.train.grpo; "
            "import fightcal.train.sft; "
            "assert not {'torch', 'trl', 'peft', 'datasets'} & sys.modules.keys()",
        ],
        env=__import__("os").environ | {"PYTHONPATH": "src"},
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
