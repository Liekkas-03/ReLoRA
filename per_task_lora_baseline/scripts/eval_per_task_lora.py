from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import GenerationCollator, get_train_eval_raw, preprocess_generation_dataset
from src.metrics import accuracy_from_texts
from src.modeling import load_base_causal_lm, load_tokenizer
from src.task_specs import get_task_spec
from src.utils import safe_mkdir, save_json, set_global_seed, split_task_arg


@torch.no_grad()
def evaluate_one(cfg, task_name: str, adapters_root: Path) -> dict:
    task = get_task_spec(task_name)
    adapter_path = adapters_root / task.name / "adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found for {task.name}: {adapter_path}")

    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "left"
    model = load_base_causal_lm(cfg)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if not (cfg.training.load_in_8bit or cfg.training.load_in_4bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    _, eval_raw = get_train_eval_raw(task, cfg.data, cache_dir=cfg.dataset_cache_dir)
    eval_ds = preprocess_generation_dataset(eval_raw, task, tokenizer, cfg.data)
    collator = GenerationCollator(tokenizer=tokenizer)
    loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    pred_texts: list[str] = []
    gold_ids: list[int] = []
    for batch in tqdm(loader, desc=f"Evaluating {task.name}"):
        gold_label = batch.pop("gold_label")
        gold_ids.extend(gold_label.tolist())
        batch = {key: value.to(device) for key, value in batch.items()}
        prompt_len = batch["input_ids"].shape[1]
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=cfg.training.generation_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        new_tokens = generated[:, prompt_len:]
        pred_texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    metrics = accuracy_from_texts(pred_texts, gold_ids, task)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--adapters_root", default=None, help="Root containing task/adapter directories.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names. Defaults to config tasks.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    adapters_root = Path(args.adapters_root) if args.adapters_root else cfg.run_output_dir
    tasks = split_task_arg(args.tasks, cfg.tasks)

    all_metrics = {}
    for task_name in tasks:
        metrics = evaluate_one(cfg, task_name, adapters_root)
        all_metrics[task_name] = metrics
        print(f"{task_name}: {metrics}")

    output = Path(args.output) if args.output else adapters_root / "per_task_eval_metrics.json"
    safe_mkdir(output.parent)
    save_json(output, all_metrics)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
