#!/usr/bin/env python3
"""Convert pseudo-user interactions into HydraLoRA SFT multiple-choice data."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version-dir",
        default="/vepfs-cnbja62d5d769987/liushaokun/sys_work/dataset_human_like/final_data_use/data/data/final_versions/v2fix_all",
        help="Directory containing pseudo_user_interactions.parquet JSONL fallback.",
    )
    parser.add_argument(
        "--output-dir",
        default="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all",
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-samples-per-user-domain", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def clean_text(value, max_chars=220):
    text = "" if value is None else str(value)
    text = "".join(ch if ord(ch) >= 32 or ch in "\t\n\r" else " " for ch in text)
    text = " ".join(text.replace("\n", " ").split())
    return text[:max_chars]


def read_interactions(path):
    groups = defaultdict(list)
    item_pool = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "__meta__" in row:
                continue
            domain = row.get("dataset", "unknown")
            item_id = str(row.get("item_id", ""))
            row["item_id"] = item_id
            row["item_text"] = clean_text(row.get("item_text") or item_id, max_chars=80)
            row["item_category"] = clean_text(row.get("item_category"), max_chars=40)
            row["timestamp"] = row.get("timestamp") or 0
            groups[(row.get("pseudo_user_id", "unknown"), domain)].append(row)
            if item_id and item_id not in item_pool[domain]:
                item_pool[domain][item_id] = row
    return groups, item_pool


def item_label(row):
    text = row.get("item_text") or row.get("item_id", "")
    category = row.get("item_category")
    if category:
        return f"{text} | Categories: {category}"
    return text


def pick_negatives(domain_items, target_id, count, rng):
    candidate_ids = [item_id for item_id in domain_items if item_id != target_id]
    if len(candidate_ids) < count:
        return []
    return [domain_items[item_id] for item_id in rng.sample(candidate_ids, count)]


def make_example(pseudo_user_id, domain, history, target, domain_items, rng, num_candidates):
    negatives = pick_negatives(domain_items, target["item_id"], num_candidates - 1, rng)
    if len(negatives) != num_candidates - 1:
        return None

    candidates = negatives + [target]
    rng.shuffle(candidates)
    answer_index = next(i for i, item in enumerate(candidates) if item["item_id"] == target["item_id"])

    history_lines = []
    for item in history:
        event = clean_text(item.get("raw_event"), max_chars=40)
        prefix = f"- [{event}] " if event else "- "
        history_lines.append(prefix + item_label(item))

    candidate_lines = [
        f"{LETTERS[i]}. {item_label(item)}" for i, item in enumerate(candidates)
    ]

    instruction = (
        "Given a pseudo-user's cross-domain interaction history, choose the most likely next item "
        "from the candidate list. Answer with only one letter: A, B, C, or D."
    )
    input_text = "\n".join(
        [
            f"Pseudo user: {pseudo_user_id}",
            f"Target domain: {domain}",
            "",
            "Recent history:",
            *history_lines,
            "",
            "Candidate items:",
            *candidate_lines,
        ]
    )
    return {
        "task_type": "pseudo_user_next_item",
        "pseudo_user_id": pseudo_user_id,
        "domain": domain,
        "target_item_id": target["item_id"],
        "instruction": instruction,
        "input": input_text,
        "output": LETTERS[answer_index],
    }


def build_examples(args):
    interactions_path = Path(args.version_dir) / "pseudo_user_interactions.parquet"
    groups, item_pool = read_interactions(interactions_path)
    rng = random.Random(args.seed)
    examples = []

    for (pseudo_user_id, domain), events in sorted(groups.items()):
        if len(events) <= args.min_history:
            continue
        events.sort(key=lambda row: (row.get("timestamp") or 0, row.get("item_id", "")))
        positions = list(range(args.min_history, len(events)))
        if args.max_samples_per_user_domain and len(positions) > args.max_samples_per_user_domain:
            step = len(positions) / float(args.max_samples_per_user_domain)
            positions = [positions[int(i * step)] for i in range(args.max_samples_per_user_domain)]

        for pos in positions:
            history_start = max(0, pos - args.history_size)
            sample_rng = random.Random(f"{args.seed}:{pseudo_user_id}:{domain}:{pos}")
            example = make_example(
                pseudo_user_id=pseudo_user_id,
                domain=domain,
                history=events[history_start:pos],
                target=events[pos],
                domain_items=item_pool[domain],
                rng=sample_rng,
                num_candidates=args.num_candidates,
            )
            if example is not None:
                examples.append(example)

    rng.shuffle(examples)
    if args.max_samples and len(examples) > args.max_samples:
        examples = examples[: args.max_samples]
    return examples


def main():
    args = parse_args()
    if args.num_candidates > len(LETTERS):
        raise ValueError("num-candidates is too large")
    if not 0.0 < args.valid_ratio < 0.5:
        raise ValueError("valid-ratio must be between 0 and 0.5")

    examples = build_examples(args)
    if not examples:
        raise RuntimeError("No SFT examples were generated")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_size = max(1, int(len(examples) * args.valid_ratio))
    valid = examples[:valid_size]
    train = examples[valid_size:]

    train_path = output_dir / "train.json"
    valid_path = output_dir / "valid.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with valid_path.open("w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} train examples to {train_path}")
    print(f"Wrote {len(valid)} valid examples to {valid_path}")
    print("First train example:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
