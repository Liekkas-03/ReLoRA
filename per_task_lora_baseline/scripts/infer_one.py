from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_chat_prompt
from src.metrics import decode_label_id
from src.modeling import load_base_causal_lm, load_tokenizer
from src.task_specs import get_task_spec


def make_example(args, task_name: str) -> dict:
    if task_name == "dbpedia":
        return {"title": args.title or "", "content": args.text or ""}
    if task_name == "amazon":
        return {"title": args.title or "", "content": args.text or ""}
    if task_name == "yelp":
        return {"text": args.text or ""}
    if task_name == "yahoo":
        return {
            "question_title": args.title or "",
            "question_content": args.text or "",
            "best_answer": args.answer or "",
        }
    if task_name == "ag_news":
        return {"text": args.text or ""}
    raise ValueError(f"Unsupported task: {task_name}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--answer", default="")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = get_task_spec(args.task)
    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "left"
    model = load_base_causal_lm(cfg)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if not (cfg.training.load_in_8bit or cfg.training.load_in_4bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    example = make_example(args, task.name)
    prompt = build_chat_prompt(tokenizer, task.build_prompt(example))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.data.max_source_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens or cfg.training.generation_max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    pred_text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
    pred_id = decode_label_id(pred_text, task)
    pred_label = task.label_names[pred_id] if pred_id >= 0 else "unparsed"

    print("Prompt:")
    print(prompt)
    print("\nRaw prediction:")
    print(pred_text)
    print("\nParsed label:")
    print(pred_label)


if __name__ == "__main__":
    main()
