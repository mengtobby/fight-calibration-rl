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


def pick_probabilities(model: Any, tok: Any, messages_batch: list) -> list[float]:
    import torch

    if not messages_batch:
        return []
    inputs = _inputs(model, tok, messages_batch, [ANSWER_PREFIX] * len(messages_batch))
    ids = [tok.encode(letter, add_special_tokens=False)[0] for letter in ("A", "B")]
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, ids].float()
        return logits.softmax(dim=-1)[:, 0].cpu().tolist()
