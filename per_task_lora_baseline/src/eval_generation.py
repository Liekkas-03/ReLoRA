from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import OLoRADecoderCollator, get_train_eval_raw
from .metrics import compute_olora_metrics, strip_answer_prefix
from .task_specs import get_task_spec
from .utils import safe_mkdir, save_json


@torch.no_grad()
def evaluate_generation_model(cfg, task_name: str, model, tokenizer, task_output_dir: Path) -> dict:
    task = get_task_spec(task_name)
    model.eval()
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
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
