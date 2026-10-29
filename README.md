# fight-calibration-rl

Train a small LM with GRPO to reason through an MMA matchup (reach, styles, recent form,
cardio) and output a win probability, rewarded with a proper scoring rule (Brier or log loss)
so it is paid for calibrated probabilities, not just correct picks. Compare against
fine-tuning on outcomes alone and against betting-market odds.

**Research question:** does RL with a probability-based reward produce better-calibrated
reasoning than fine-tuning on outcomes alone?

**Reward-hacking angle:** does the model learn to always predict close to 50%, or to ignore its
own reasoning and copy the odds?

See [docs/research-plan.md](docs/research-plan.md) for the design, arms, metrics and diagnostics.

## Layout

```
src/fightcal/   package (data, prompts, rewards, metrics, diagnostics, training, eval)
scripts/        CLI entry points (prepare data, train, predict, report)
configs/        experiment configs, one per arm
notebooks/      Kaggle/Colab runner
tests/          CPU-only tests (no model weights needed)
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest -q            # CPU-only; model tests are marked `model` and skipped by default
ruff check src tests scripts
```

Training needs a GPU and the `model` extra (`pip install -e ".[model]"`); see
[docs/cloud.md](docs/cloud.md) for running it on a free Kaggle or Colab GPU.
