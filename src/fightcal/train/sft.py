"""Optional GPU SFT training; heavyweight imports stay inside functions."""

import json
from pathlib import Path
from typing import Any

from fightcal.data.io import load_bouts_jsonl
from fightcal.prompts import PromptOptions
from fightcal.train.config import SFTRunConfig
from fightcal.train.records import select_subset, sft_records


def _trainer_config(cfg: SFTRunConfig, bf16: bool) -> Any:
    from trl import SFTConfig

    return SFTConfig(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        model_init_kwargs={"torch_dtype": "bfloat16" if bf16 else "float16", "use_cache": False},
        report_to="none",
        eval_strategy="no",
        max_length=cfg.max_seq_length,
        num_train_epochs=cfg.num_epochs,
    )


def run_sft(cfg: SFTRunConfig) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import SFTTrainer

    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    opts = PromptOptions(anonymize=cfg.anonymize, include_odds=cfg.include_odds)
    bouts = select_subset(load_bouts_jsonl(cfg.train_path), cfg.max_train_bouts, cfg.seed)
    validation = load_bouts_jsonl(cfg.val_path)
    if not bouts or not validation:
        raise ValueError("training and validation datasets must be nonempty")
    train = Dataset.from_list(sft_records(bouts, opts))
    val = Dataset.from_list(sft_records(validation, opts))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=cfg.model_name,
        args=_trainer_config(cfg, torch.cuda.is_bf16_supported()),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        ),
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "fightcal_run.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return str(output)
