"""Append resumable model predictions and diagnostic interventions."""

import argparse
from dataclasses import replace
from pathlib import Path

from fightcal.data.io import load_bouts_jsonl
from fightcal.data.odds import market_prob_a
from fightcal.infer import generate_completions, load_model, pick_probabilities
from fightcal.predictions import PredictionRecord, read_predictions, write_predictions
from fightcal.prompts import build_messages, parse_probability
from fightcal.variants import plan_variants, to_original_perspective


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arm", "model", "bouts", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--mode", choices=["reason_prob", "pick"], default="reason_prob")
    parser.add_argument("--variants")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.n_samples < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch size, samples and limit must be positive")
    bouts = load_bouts_jsonl(args.bouts)
    if args.limit is not None:
        bouts = bouts[: args.limit]
    lookup = {b.bout_id: b for b in bouts}
    names = (
        args.variants
        or ("base,swapped,odds_in_prompt,odds_perturbed" if "odds" in args.arm else "base,swapped")
    ).split(",")
    names = list(dict.fromkeys(n.strip() for n in names))
    if args.mode == "pick" and "reasoning_swap" in names:
        raise ValueError("reasoning_swap requires reason_prob mode")
    if "reasoning_swap" in names and "base" not in names:
        names.insert(0, "base")
    names = [n for n in names if n != "reasoning_swap"] + (
        ["reasoning_swap"] if "reasoning_swap" in names else []
    )
    existing = read_predictions(args.out) if args.resume and Path(args.out).exists() else []
    if any(r.arm != args.arm or r.checkpoint != args.checkpoint for r in existing):
        raise ValueError("resume output must contain only the requested arm and checkpoint")
    done = {(r.bout_id, r.variant) for r in existing}
    base = {
        r.bout_id: r.completion
        for r in existing
        if r.variant == "base" and r.completion is not None and r.sample_idx == 0
    }
    if not args.resume:
        write_predictions(args.out, [])
    model, tok = load_model(args.model, args.adapter)
    for name in names:
        requests = plan_variants(bouts, [name], base_completions=base, seed=args.seed)
        requests = [r for r in requests if (r.bout_id, r.variant) not in done]
        for start in range(0, len(requests), args.batch_size):
            batch = requests[start : start + args.batch_size]
            messages = [
                build_messages(
                    lookup[r.bout_id].swapped() if r.swapped else lookup[r.bout_id],
                    replace(r.opts, mode=args.mode),
                )
                for r in batch
            ]
            if args.mode == "pick":
                probabilities = pick_probabilities(model, tok, messages)
                completions = [[None] for _ in batch]
            else:
                probabilities = []
                completions = generate_completions(
                    model,
                    tok,
                    messages,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    forced_prefixes=[r.forced_prefix for r in batch],
                )
            rows = []
            for index, (request, texts) in enumerate(zip(batch, completions, strict=True)):
                bout = lookup[request.bout_id]
                market = market_prob_a(bout)
                shown = request.opts.odds_override_prob_a
                if shown is None and request.opts.include_odds:
                    shown = market
                for sample, text in enumerate(texts):
                    p = probabilities[index] if args.mode == "pick" else parse_probability(text)
                    rows.append(
                        PredictionRecord(
                            bout.bout_id,
                            args.arm,
                            name,
                            sample,
                            to_original_perspective(p, request.swapped),
                            bout.a_won,
                            market,
                            shown,
                            text,
                            args.checkpoint,
                        )
                    )
                if name == "base" and texts[0] is not None:
                    base[bout.bout_id] = texts[0]
            write_predictions(args.out, rows, append=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
