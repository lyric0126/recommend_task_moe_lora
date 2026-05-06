#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from peft import PeftModel


PROMPT_TEMPLATE = "{instruction}</s>"
CHOICES = ["A", "B", "C", "D"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--group-field", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def load_examples(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit and limit > 0:
        data = data[:limit]
    return data


def build_source(example):
    instruction = example["instruction"]
    input_text = example.get("input") or ""
    if input_text:
        instruction = instruction + "\n" + input_text
    return PROMPT_TEMPLATE.format_map({"instruction": instruction})


def encode_candidate(tokenizer, source, choice, max_seq_length):
    target = f"{choice}{tokenizer.eos_token}"
    source_ids = tokenizer(source, return_attention_mask=False)["input_ids"]
    target_ids = tokenizer(
        target,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]
    if len(target_ids) >= max_seq_length:
        source_ids = []
        target_ids = target_ids[:max_seq_length]
    else:
        source_ids = source_ids[-(max_seq_length - len(target_ids)) :]
    input_ids = torch.tensor(source_ids + target_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(source_ids) + target_ids, dtype=torch.long)
    letter_labels = torch.tensor([-100] * len(source_ids) + target_ids[:1] + [-100] * (len(target_ids) - 1), dtype=torch.long)
    return input_ids, labels, letter_labels


def score_batch(model, tokenizer, encoded, device):
    input_ids = [x[0] for x in encoded]
    labels = [x[1] for x in encoded]
    letter_labels = [x[2] for x in encoded]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
    letter_labels = pad_sequence(letter_labels, batch_first=True, padding_value=-100).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)

    def gather_scores(label_tensor):
        shifted = label_tensor[:, 1:]
        mask = shifted.ne(-100)
        safe = shifted.masked_fill(~mask, 0)
        token_scores = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        return (token_scores * mask).sum(dim=-1).detach().cpu()

    return gather_scores(labels), gather_scores(letter_labels)


def main():
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    if device.startswith("cuda"):
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    model.eval()

    examples = load_examples(data_path, args.limit)
    total = 0
    correct_full = 0
    correct_letter = 0
    groups = defaultdict(lambda: {"total": 0, "correct_full": 0, "correct_letter": 0})
    rows = []

    flat_encoded = []
    flat_meta = []
    for idx, ex in enumerate(examples):
        answer = str(ex["output"]).strip()[:1].upper()
        if answer not in CHOICES:
            continue
        source = build_source(ex)
        for choice in CHOICES:
            flat_encoded.append(encode_candidate(tokenizer, source, choice, args.max_seq_length))
            flat_meta.append((idx, choice, answer))

    per_example = {}
    step = args.batch_size * len(CHOICES)
    for start in tqdm(range(0, len(flat_encoded), step), desc="scoring"):
        encoded = flat_encoded[start : start + step]
        meta = flat_meta[start : start + step]
        full_scores, letter_scores = score_batch(model, tokenizer, encoded, device)
        for (idx, choice, answer), full_score, letter_score in zip(meta, full_scores.tolist(), letter_scores.tolist()):
            rec = per_example.setdefault(idx, {"answer": answer, "full": {}, "letter": {}})
            rec["full"][choice] = full_score
            rec["letter"][choice] = letter_score

    for idx, rec in sorted(per_example.items()):
        ex = examples[idx]
        pred_full = max(CHOICES, key=lambda c: rec["full"].get(c, float("-inf")))
        pred_letter = max(CHOICES, key=lambda c: rec["letter"].get(c, float("-inf")))
        answer = rec["answer"]
        group = str(ex.get(args.group_field, "all")) if args.group_field else "all"

        total += 1
        correct_full += int(pred_full == answer)
        correct_letter += int(pred_letter == answer)
        groups[group]["total"] += 1
        groups[group]["correct_full"] += int(pred_full == answer)
        groups[group]["correct_letter"] += int(pred_letter == answer)
        rows.append(
            {
                "index": idx,
                "answer": answer,
                "pred_full": pred_full,
                "pred_letter": pred_letter,
                "group": group,
                "scores_full": rec["full"],
                "scores_letter": rec["letter"],
            }
        )

    result = {
        "data": str(data_path),
        "adapter": str(args.adapter),
        "base_model": str(args.base_model),
        "total": total,
        "accuracy_full_target": correct_full / total if total else 0.0,
        "accuracy_letter_only": correct_letter / total if total else 0.0,
        "group_field": args.group_field or None,
        "groups": {
            group: {
                "total": stat["total"],
                "accuracy_full_target": stat["correct_full"] / stat["total"] if stat["total"] else 0.0,
                "accuracy_letter_only": stat["correct_letter"] / stat["total"] if stat["total"] else 0.0,
            }
            for group, stat in sorted(groups.items())
        },
        "predictions": rows,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
