# Research plan: calibrated reasoning RL for fight prediction

## Question

Does RL with a probability-based (proper scoring rule) reward produce better-calibrated
reasoning than fine-tuning on outcomes alone? And how close does a small reasoning model get to
betting markets, which aggregate far more information than a stat sheet?

## Setup

- **Data.** Historical UFC bouts with pre-fight fighter stats and closing moneyline odds
  (Kaggle "Ultimate UFC Dataset", `ufc-master.csv`; both the older `R_*/B_*` and the newer
  `Red*/Blue*` column layouts are supported). Draws and no-contests are dropped.
- **Split.** Strictly temporal (train < val < test by date), so no future information leaks
  into training. Red corner is usually the favorite, so corners are randomly reassigned to
  "Fighter A"/"Fighter B" (seeded per bout) to stop the model learning a position prior.
- **Prompt.** A matchup card: age, height, reach, stance, record, streaks, recent form, striking
  and grappling rates, cardio proxies (rounds fought, decision wins), scheduled rounds.
  Fighters are anonymized by default, because a pretrained LM may remember the result of a
  famous fight by name. Odds are shown only in the odds-conditioned arms.
- **Output.** `<think>…</think><answer>0.63</answer>` where 0.63 is P(Fighter A wins).
- **Model.** Qwen2.5-1.5B-Instruct + LoRA, trained on a free Kaggle/Colab GPU.

## Arms

| Arm | Training signal | Odds in prompt |
|---|---|---|
| `base` | none (zero-shot reasoning) | no |
| `sft_outcome` | SFT on the winner label; probability read from P(A)/(P(A)+P(B)) logits | no |
| `grpo_correct` | GRPO, reward 1 if the pick is right (improper: ignores confidence) | no |
| `grpo_brier` | GRPO, reward 1 − (p − y)² | no |
| `grpo_log` | GRPO, reward log p(y), clipped | no |
| `grpo_brier_odds` | GRPO, Brier reward | yes |

Non-LM baselines: de-vigged market probability, logistic regression on stat differences,
constant base rate, constant 0.5.

Note: SFT with cross-entropy on the label token is itself log-loss training of the token
distribution, so `sft_outcome` is a strong calibration baseline, not a strawman. The comparison
that isolates "reasoning + verbalized probability" is `grpo_brier`/`grpo_log` vs `grpo_correct`
(same RL machinery, proper vs improper reward) and vs `sft_outcome` (no reasoning).

## Metrics

Brier, log loss, accuracy, ECE (uniform and quantile bins), reliability diagram, Murphy
decomposition (reliability / resolution / uncertainty), sharpness. Paired bootstrap CIs for
every arm vs the market, and a forecast-encompassing regression
(y ~ logit p_market + logit p_model) to test whether the model adds information beyond the odds.

## Reward-hacking diagnostics

1. **Collapse to 50%.** A proper scoring rule pays a constant 0.5 a guaranteed Brier of 0.25. If
   the model can't find signal it can retreat there. Track the share of predictions within ±0.05
   of 0.5, mean |p − 0.5|, and resolution over training checkpoints (parsed probabilities are
   logged during GRPO).
2. **Copying the odds.** In the odds-conditioned arm, measure the correlation and mean absolute
   gap between model and market, and the slope of logit p_model on logit p_market. Counterfactual:
   re-run with perturbed odds in the prompt and measure dp/dp_market. A slope near 1 with the
   reasoning unchanged means the model is copying, not reasoning.
3. **Ignoring its own reasoning.** Swap in the reasoning from a different bout before the answer
   and measure how much p moves. Also check that the fighter the reasoning favors agrees with
   the side of 0.5 the answer is on.
4. **Symmetry.** Swap A and B: a coherent forecaster has p(A beats B) + p(B beats A) = 1.
5. **Format gaming.** Unparseable answers get a reward strictly below the worst valid one, so
   refusing to answer never pays. Parse rate is reported per arm.

## Expected outcomes and what would be interesting

- Proper-reward GRPO beats `grpo_correct` on calibration (ECE, reliability) at similar accuracy.
- All LM arms lose to the market on Brier; the encompassing test says whether any of them add
  information on top of it.
- Collapse is likely early in training for `grpo_log` (log loss punishes confident errors
  hardest); this is measurable from the logged training probabilities.
