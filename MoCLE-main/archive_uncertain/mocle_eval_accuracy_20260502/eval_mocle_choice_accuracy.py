#!/usr/bin/env python3
"""Evaluate MoCLE adapters with 4-choice A/B/C/D accuracy."""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


PROMPT_TEMPLATE = "{instruction}</s>"
CHOICES = ["A", "B", "C", "D"]
DOMAIN_TO_EXPERT = {
    "movielens": "expert_0",
    "goodreads": "expert_1",
    "mind": "expert_2",
    "kuairec": "expert_3",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True, help="MoCLE checkpoint directory containing expert_*/ adapters.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of examples per scoring batch.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--group-field", default="domain")
    parser.add_argument("--route-field", default="domain")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def ensure_local_peft():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    peft_src = os.path.join(repo_root, "peft-main", "src")
    if peft_src not in sys.path:
        sys.path.insert(0, peft_src)


def load_examples(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit and limit > 0 else data


def build_source(example):
    instruction = example["instruction"]
    input_text = example.get("input") or ""
    if input_text:
        instruction = instruction + "\n" + input_text
    return PROMPT_TEMPLATE.format_map({"instruction": instruction})


def encode_candidate(tokenizer, source, choice, max_seq_length):
    target = "{}{}".format(choice, tokenizer.eos_token)
    source_ids = tokenizer(source, return_attention_mask=False)["input_ids"]
    target_ids = tokenizer(target, return_attention_mask=False, add_special_tokens=False)["input_ids"]
    if len(target_ids) >= max_seq_length:
        source_ids = []
        target_ids = target_ids[:max_seq_length]
    else:
        source_ids = source_ids[-(max_seq_length - len(target_ids)) :]
    input_ids = torch.tensor(source_ids + target_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(source_ids) + target_ids, dtype=torch.long)
    letter_labels = torch.tensor(
        [-100] * len(source_ids) + target_ids[:1] + [-100] * (len(target_ids) - 1),
        dtype=torch.long,
    )
    return input_ids, labels, letter_labels


def route_to_expert(example, route_field):
    value = str(example.get(route_field, "")).lower()
    if value in DOMAIN_TO_EXPERT:
        return DOMAIN_TO_EXPERT[value]
    cluster_id = example.get("cluster_id")
    if cluster_id is None:
        cluster_id = example.get("task_type", 0)
    try:
        return "expert_{}".format(int(cluster_id) % len(DOMAIN_TO_EXPERT))
    except Exception:
        return "expert_0"


def score_batch(model, tokenizer, encoded, device):
    input_ids = [x[0] for x in encoded]
    labels = [x[1] for x in encoded]
    letter_labels = [x[2] for x in encoded]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
    letter_labels = pad_sequence(letter_labels, batch_first=True, padding_value=-100).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]

    def gather_scores(label_tensor):
        shifted = label_tensor[:, 1:]
        mask = shifted.ne(-100)
        if not mask.any():
            return torch.zeros(shifted.shape[0], dtype=torch.float32)
        selected_logits = logits[mask].float()
        selected_labels = shifted[mask]
        selected_scores = torch.log_softmax(selected_logits, dim=-1).gather(
            -1, selected_labels.unsqueeze(-1)
        ).squeeze(-1)
        row_ids = torch.arange(shifted.shape[0], device=shifted.device).unsqueeze(1).expand_as(shifted)[mask]
        scores = torch.zeros(shifted.shape[0], dtype=torch.float32, device=shifted.device)
        scores.index_add_(0, row_ids, selected_scores)
        return scores.detach().cpu()

    return gather_scores(labels), gather_scores(letter_labels)


def load_mocle_model(args, dtype, device):
    ensure_local_peft()
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    if hasattr(model, "config"):
        model.config.use_cache = False
    first_adapter = os.path.join(args.checkpoint, "expert_0")
    model = PeftModel.from_pretrained(model, first_adapter, adapter_name="expert_0")
    for expert_idx in range(1, len(DOMAIN_TO_EXPERT)):
        name = "expert_{}".format(expert_idx)
        model.load_adapter(os.path.join(args.checkpoint, name), adapter_name=name)
    if device.startswith("cuda"):
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_mocle_model(args, dtype, device)
    examples = load_examples(args.data, args.limit)

    encoded_by_expert = defaultdict(list)
    meta_by_expert = defaultdict(list)
    for idx, ex in enumerate(examples):
        answer = str(ex["output"]).strip()[:1].upper()
        if answer not in CHOICES:
            continue
        source = build_source(ex)
        expert = route_to_expert(ex, args.route_field)
        for choice in CHOICES:
            encoded_by_expert[expert].append(encode_candidate(tokenizer, source, choice, args.max_seq_length))
            meta_by_expert[expert].append((idx, choice, answer, expert))

    per_example = {}
    for expert in sorted(encoded_by_expert):
        model.set_adapter(expert)
        step = args.batch_size * len(CHOICES)
        encoded_items = encoded_by_expert[expert]
        meta_items = meta_by_expert[expert]
        for start in tqdm(range(0, len(encoded_items), step), desc="scoring {}".format(expert)):
            encoded = encoded_items[start : start + step]
            meta = meta_items[start : start + step]
            full_scores, letter_scores = score_batch(model, tokenizer, encoded, device)
            for (idx, choice, answer, expert_name), full_score, letter_score in zip(
                meta, full_scores.tolist(), letter_scores.tolist()
            ):
                rec = per_example.setdefault(idx, {"answer": answer, "full": {}, "letter": {}, "expert": expert_name})
                rec["full"][choice] = full_score
                rec["letter"][choice] = letter_score

    total = 0
    correct_full = 0
    correct_letter = 0
    groups = defaultdict(lambda: {"total": 0, "correct_full": 0, "correct_letter": 0})
    rows = []
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
                "expert": rec["expert"],
                "scores_full": rec["full"],
                "scores_letter": rec["letter"],
            }
        )

    result = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "total": total,
        "accuracy_full_target": correct_full / total if total else 0.0,
        "accuracy_letter_only": correct_letter / total if total else 0.0,
        "group_field": args.group_field or None,
        "route_field": args.route_field,
        "domain_to_expert": DOMAIN_TO_EXPERT,
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
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
