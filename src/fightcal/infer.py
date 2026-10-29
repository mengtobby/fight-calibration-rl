"""GPU inference helpers; heavyweight dependencies are loaded only on demand."""

from typing import Any

from fightcal.prompts import ANSWER_PREFIX


def load_model(model_name: str, adapter_dir: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def _inputs(model: Any, tok: Any, messages_batch: list, prefixes: list[str]) -> Any:
    tok.padding_side = "left"
    prompts = [
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
        for messages, prefix in zip(messages_batch, prefixes, strict=True)
    ]
    return tok(prompts, return_tensors="pt", padding=True).to(model.device)


def generate_completions(
    model: Any,
    tok: Any,
    messages_batch: list,
    *,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    forced_prefixes: list[str | None] | None = None,
) -> list[list[str]]:
    import torch

    if n_samples < 1 or max_new_tokens < 1 or temperature < 0:
        raise ValueError(
            "samples and token limit must be positive; temperature must be nonnegative"
        )
    prefixes = (
        [p or "" for p in forced_prefixes]
        if forced_prefixes is not None
        else ["" for _ in messages_batch]
    )
    if len(prefixes) != len(messages_batch):
        raise ValueError("forced_prefixes must match batch length")
    if not messages_batch:
        return []
    inputs = _inputs(model, tok, messages_batch, prefixes)
    result: list[list[str]] = [[] for _ in messages_batch]
    for _ in range(n_samples):
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature} if temperature > 0 else {}),
                pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for i, text in enumerate(texts):
            result[i].append(prefixes[i] + text)
    return result


def _pick_token_context(tok: Any, messages: list) -> tuple[list[int], tuple[int, int]]:
    """Find the label decision in jointly tokenized, training-format answers.

    For Qwen the differing tokens are typically '>A' and '>B', so the shared
    context ends at '<answer', before the closing angle bracket.
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answers = [
        tok.encode(prompt + ANSWER_PREFIX + letter + "</answer>", add_special_tokens=False)
        for letter in ("A", "B")
    ]
    for index, (a, b) in enumerate(zip(*answers, strict=False)):
        if a != b:
            if index == 0:
                raise ValueError("answer tokenizations must have a nonempty shared context")
            return answers[0][:index], (a, b)
    raise ValueError("A and B answers must have distinct tokenizations")


def pick_probabilities(model: Any, tok: Any, messages_batch: list) -> list[float]:
    """Normalize logits at the first differing token of the trained A/B answers."""
    import torch

    if not messages_batch:
        return []
    contexts = [_pick_token_context(tok, messages) for messages in messages_batch]
    tok.padding_side = "left"
    inputs = tok.pad(
        [{"input_ids": context, "attention_mask": [1] * len(context)} for context, _ in contexts],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    ids = torch.tensor([pair for _, pair in contexts], device=model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :].float().gather(1, ids)
        return logits.softmax(dim=-1)[:, 0].cpu().tolist()
