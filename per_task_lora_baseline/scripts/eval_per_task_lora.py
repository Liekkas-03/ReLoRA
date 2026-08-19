from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import OLoRADecoderCollator, get_train_eval_raw
from src.metrics import compute_olora_metrics, strip_answer_prefix
from src.modeling import load_base_causal_lm, load_tokenizer
from src.task_specs import get_task_spec
from src.utils import safe_mkdir, save_json, set_global_seed, split_task_arg


@torch.no_grad()
def evaluate_one(cfg, task_name: str, adapter_path: Path, task_output_dir: Path) -> dict:
    task = get_task_spec(task_name)
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

    _, eval_ds, _ = get_train_eval_raw(task, cfg.data)
    collator = OLoRADecoderCollator(tokenizer=tokenizer, data_cfg=cfg.data, train=False)
    loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    pred_texts: list[str] = []
    gold_labels: list[str] = []
    prediction_rows: list[dict] = []
    for batch in tqdm(loader, desc=f"Evaluating {task.name}"):
        metadata = batch.pop("metadata")
        batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=cfg.training.generation_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for text, meta in zip(decoded, metadata):
            prediction = strip_answer_prefix(text)
            gold_label = meta["Instance"]["label"]
            pred_texts.append(prediction)
            gold_labels.append(gold_label)
            prediction_rows.append(
                {
                    "Task": meta["Task"],
                    "Dataset": meta["Dataset"],
                    "Instance": meta["Instance"],
                    "Prediction": prediction,
                }
            )

    raw_metrics = compute_olora_metrics(pred_texts, gold_labels)
    metrics = {
        "predict_exact_match": raw_metrics["exact_match"],
        "predict_rouge1": raw_metrics["rouge1"],
        "predict_rougeL": raw_metrics["rougeL"],
        "predict_samples": len(gold_labels),
    }
    safe_mkdir(task_output_dir)
    predictions_path = task_output_dir / "predict_eval_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(task_output_dir / "predict_results.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--adapters_root", default=None, help="Root containing task/adapter directories.")
    parser.add_argument("--single_adapter_path", default=None, help="Use one adapter to evaluate every task.")
    parser.add_argument("--single_output_root", default=None, help="Output root for --single_adapter_path results.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names. Defaults to config tasks.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    adapters_root = Path(args.adapters_root) if args.adapters_root else cfg.run_output_dir
    tasks = split_task_arg(args.tasks, cfg.tasks)

    single_adapter_path = Path(args.single_adapter_path) if args.single_adapter_path else None
    single_output_root = None
    if single_adapter_path is not None:
        single_output_root = (
            Path(args.single_output_root)
            if args.single_output_root
            else single_adapter_path.parent / "eval"
        )

    all_metrics = {}
    for task_name in tasks:
        task = get_task_spec(task_name)
        if single_adapter_path is not None:
            adapter_path = single_adapter_path
            task_output_dir = single_output_root / task.name
        else:
            adapter_path = adapters_root / task.name / "adapter"
            task_output_dir = adapters_root / task.name
        metrics = evaluate_one(cfg, task_name, adapter_path, task_output_dir)
        all_metrics[task_name] = metrics
        print(f"{task_name}: {metrics}")

    if args.output:
        output = Path(args.output)
    elif single_output_root is not None:
        output = single_output_root / "merged_eval_metrics.json"
    else:
        output = adapters_root / "per_task_eval_metrics.json"
    safe_mkdir(output.parent)
    save_json(output, all_metrics)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
