# Run fightcal on a free cloud GPU

All data preparation, installation, training, inference and reporting run in the cloud;
nothing runs locally. Open `notebooks/kaggle_run.ipynb` in a new Kaggle notebook.

1. Add Input: attach **Ultimate UFC Dataset** by **mdabbert**, containing `ufc-master.csv`.
2. In notebook Settings > Accelerator choose GPU (T4 preferred; P100 also works).
   Enable Settings > Internet. Kaggle requires phone verification for internet access.
3. Run the installation cell: it clones the repository and installs `.[model]`.
4. Run preparation and baselines. The runner locates the attached CSV under `/kaggle/input`,
   requires valid odds, randomizes corners deterministically and writes temporal splits to
   `data/processed` in the cloned repository. Inspect `summary.json` for counts and dates.
5. Edit `RUN_ARMS`. Start with `['base']`; each arm can run in a separate session.
   Available training arms are `sft_outcome`, `grpo_correct`, `grpo_brier`, `grpo_log`,
   and `grpo_brier_odds`. Each uses its matching config in `configs`.
6. Run training and prediction, then the report cell. Download `/kaggle/working/runs` before
   ending the session or save a notebook version with outputs. Adapters/checkpoints are in
   `runs/<arm>`, predictions in `runs/<arm>.jsonl`, and reports in `runs/report/report.md`
   and `report.json`. Combine prediction files from separate sessions before reporting.

Allow roughly 20-90 minutes for base inference, 30-90 minutes for SFT training,
and 2-6 hours per 300-step GRPO arm, plus inference. These are planning estimates,
not benchmarks; GPU allocation, token lengths and free quotas vary. Try `--limit 20`
with prediction and `--max-steps 5` with training for an initial smoke run. Keep smoke
outputs separate from final experiment outputs. Reduce inference batch size for memory
pressure; GRPO training batch sizes must remain divisible by the number of generations.

Prediction uses `--resume` and appends completed batches, skipping existing
`(bout_id, variant)` pairs. Restore the JSONL file to the same output path to continue.
Use one arm/checkpoint per output file. Reasoning swaps automatically obtain base
completions first; use at least two bouts. The pick-mode SFT arm uses token probabilities
and does not support reasoning swaps. The current training CLI does not expose checkpoint resume. Preserve finished adapters;
restart an interrupted training arm in a fresh output directory. Prediction resume does
not retrain or restore an interrupted optimizer.

For Colab, upload the notebook, select Runtime > Change runtime type > T4 GPU,
and upload `ufc-master.csv` into `/content` before running the data cell. The same
notebook uses `/content/runs`. Download outputs or copy them to mounted Drive before
runtime expiry; Colab storage is temporary too. No paid GPU service is required.

Odds-conditioned arms keep odds in their base, swapped and reasoning-swap prompts.
The runner passes `--include-odds` for these arms; the prediction CLI also recognizes
odds arm names. For a custom odds-conditioned arm, pass `--include-odds` explicitly.
Swapped prompts use the swapped fighters' own odds. Pick-mode probabilities normalize
logits at the first differing token of the jointly tokenized full A/B answers, matching
the SFT answer format (including tokenizers that fuse the angle bracket and letter).
Regenerate prediction files made before these fixes; `--resume` preserves existing rows.
