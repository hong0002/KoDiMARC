from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, default_data_collator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kodimarc.step1.dataset import ConnectiveSFTConfig, ConnectiveSFTDataset
from kodimarc.step1.modeling import build_plain_sft_model



def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def patch_config_for_json(config):
    def safe_to_json_string(*args, **kwargs):
        return "{}\n"

    config.to_json_string = safe_to_json_string.__get__(config, type(config))



def main():
    parser = argparse.ArgumentParser(description="Train the Step1 discourse marker generator with response-only SFT.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "step1" / "step1_sft_example.yaml"),
        help="Path to a YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    base_out_dir = Path(train_cfg["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)
    model_name_short = model_cfg["base_model"].split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_dir / f"{timestamp}_{model_name_short}_step1_sft"
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(args.config).resolve(), run_dir / "config_used.yaml")
    shutil.copy2(Path(__file__).resolve(), run_dir / Path(__file__).name)

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = build_plain_sft_model(model_cfg, lora_cfg)
    if hasattr(model, "config"):
        patch_config_for_json(model.config)

    train_dataset = ConnectiveSFTDataset(
        ConnectiveSFTConfig(
            train_path=data_cfg["train_path"],
            max_length=data_cfg.get("max_length", 512),
            max_samples=data_cfg.get("max_samples"),
        ),
        tokenizer,
    )

    valid_path = data_cfg.get("valid_path")
    eval_dataset = None
    if valid_path:
        eval_dataset = ConnectiveSFTDataset(
            ConnectiveSFTConfig(
                train_path=valid_path,
                max_length=data_cfg.get("max_length", 512),
                max_samples=None,
            ),
            tokenizer,
        )
        print(f"[data] train={len(train_dataset):,}, valid={len(eval_dataset):,}")
    else:
        print(f"[data] train={len(train_dataset):,}, valid=None")

    logging_dir = (run_dir / train_cfg.get("logging_dir", "logs")).resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)

    quant_type = (model_cfg.get("quant_type") or "").lower()
    dtype_str = (model_cfg.get("dtype") or "").lower()
    is_quantized = quant_type in ["8bit", "4bit"]
    if is_quantized:
        use_fp16 = False
        use_bf16 = False
    else:
        use_fp16 = "16" in dtype_str
        use_bf16 = "bf16" in dtype_str

    do_eval = eval_dataset is not None
    eval_steps = train_cfg.get("eval_steps", train_cfg.get("save_steps", 500))
    save_steps = train_cfg.get("save_steps", 500)
    load_best = bool(do_eval and (not is_quantized) and train_cfg.get("load_best_model_at_end", True))

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=train_cfg.get("num_epochs", 2),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 1),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        learning_rate=train_cfg.get("lr", 1e-5),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg.get("logging_steps", 20),
        fp16=use_fp16,
        bf16=use_bf16,
        seed=seed,
        save_safetensors=False,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=False,
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
    )

    callbacks = []
    if do_eval:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
                early_stopping_threshold=train_cfg.get("early_stopping_threshold", 0.0),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    print("==== Step1 response-only SFT training start ====")
    trainer.train()
    trainer.save_model(str(run_dir))
    print(f"==== Training done. Saved to {run_dir} ====")


if __name__ == "__main__":
    main()
