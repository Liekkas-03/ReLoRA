from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path

from peft import PeftModel
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import OLoRADecoderCollator, get_train_eval_raw
from src.modeling import attach_new_lora, load_base_causal_lm, load_tokenizer
from src.task_specs import get_task_spec
from src.utils import safe_mkdir, save_json, set_global_seed, split_task_arg


def build_training_args(cfg, task_output_dir: Path):
    train_cfg = cfg.training
    kwargs = dict(
        output_dir=str(task_output_dir / "trainer"),
        overwrite_output_dir=train_cfg.overwrite_output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        save_total_limit=train_cfg.save_total_limit,
        report_to=[] if train_cfg.report_to == "none" else train_cfg.report_to,
        remove_unused_columns=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = train_cfg.eval_strategy
    else:
        kwargs["evaluation_strategy"] = train_cfg.eval_strategy
    if "optim" in sig.parameters:
        kwargs["optim"] = train_cfg.optimizer
    if "lr_scheduler_type" in sig.parameters:
        kwargs["lr_scheduler_type"] = train_cfg.lr_scheduler_type
    if "data_seed" in sig.parameters:
        kwargs["data_seed"] = cfg.data.data_seed
    if "tf32" in sig.parameters:
        kwargs["tf32"] = train_cfg.tf32
    return TrainingArguments(**kwargs)


def train_one_task(cfg, task_name: str) -> None:
    task = get_task_spec(task_name)
    task_output_dir = cfg.run_output_dir / task.name
    adapter_dir = task_output_dir / "adapter"

    if adapter_dir.exists() and not cfg.training.overwrite_output_dir:
        print(f"[skip] {task.name}: adapter already exists at {adapter_dir}")
        return
    if task_output_dir.exists() and cfg.training.overwrite_output_dir:
        shutil.rmtree(task_output_dir)
    safe_mkdir(task_output_dir)

    print(f"\n===== Training independent LoRA for task: {task.name} =====")
    print(f"Base model: {cfg.base_model_name_or_path}")
    print("This task starts from the original base model, not from a previous adapter.")

    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "right"
    model = load_base_causal_lm(cfg)
    model = attach_new_lora(model, cfg)

    train_raw, eval_raw, labels = get_train_eval_raw(task, cfg.data)
    data_collator = OLoRADecoderCollator(tokenizer=tokenizer, data_cfg=cfg.data, train=True)

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, task_output_dir),
        train_dataset=train_raw,
        eval_dataset=eval_raw,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_json(task_output_dir / "train_metrics.json", train_result.metrics)
    save_json(
        adapter_dir / "task_info.json",
        {
            "task": task.name,
            "task_type": task.task_type,
            "dataset_name": task.dataset_name,
            "base_model_name_or_path": cfg.base_model_name_or_path,
            "labels": labels,
            "benchmark_data_dir": cfg.data.benchmark_data_dir,
            "note": "Independent per-task LoRA adapter trained from the original base model using O-LoRA CL_Benchmark format.",
        },
    )

    print(f"[done] {task.name}: adapter saved to {adapter_dir}")

    if isinstance(model, PeftModel):
        model.unload()
    del model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names. Defaults to config tasks.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    safe_mkdir(cfg.run_output_dir)

    tasks = split_task_arg(args.tasks, cfg.tasks)
    print(f"Run output: {cfg.run_output_dir}")
    print(f"Tasks: {', '.join(tasks)}")
    for task_name in tasks:
        train_one_task(cfg, task_name)


if __name__ == "__main__":
    main()
