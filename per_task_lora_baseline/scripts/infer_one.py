from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_instance_instruction, build_model_prompt, load_task_labels
from src.metrics import exact_match_score, strip_answer_prefix
from src.modeling import load_base_causal_lm, load_tokenizer
from src.task_specs import get_task_spec


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--sentence", default=None, help="Official O-LoRA style input sentence.")
    parser.add_argument("--text", default=None, help="Alias for --sentence.")
    parser.add_argument("--label", default=None, help="Optional gold label for one-sample correctness check.")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = get_task_spec(args.task)
    sentence = args.sentence if args.sentence is not None else args.text
    if not sentence:
        raise ValueError("Please provide --sentence or --text.")

    labels = load_task_labels(task, cfg.data)
    labels_str = ", ".join(labels)
    example = {
        "Task": task.task_type,
        "Dataset": task.dataset_name,
        "Samples": [],
        "subset": "test",
        "Instance": {
            "id": "0",
            "sentence": sentence,
            "label": args.label or "",
            "ground_truth": args.label or "",
            "instruction": task.instruction + "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:",
        },
    }
    instruction = build_instance_instruction(example, cfg.data)

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

    prompt = build_model_prompt(tokenizer, instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.data.max_source_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens or cfg.training.generation_max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    decoded = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    prediction = strip_answer_prefix(decoded)

    print("Instruction:")
    print(instruction)
    print("\nAllowed labels:")
    print(", ".join(labels))
    print("\nRaw generation:")
    print(decoded)
    print("\nPrediction:")
    print(prediction)
    if args.label is not None:
        print("\nGold label:")
        print(args.label)
        print("\nExact match:")
        print(exact_match_score(prediction, args.label))


if __name__ == "__main__":
    main()
