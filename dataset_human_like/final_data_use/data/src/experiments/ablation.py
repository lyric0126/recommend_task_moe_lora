import os
from collections import Counter, defaultdict

from src.experiments.baselines import evaluate_model, train_model
import json

from src.io_utils import read_table, write_json, write_markdown


def _heldout(config, target):
    out = set()
    split_dir = os.path.join(config["paths"]["exp_splits"], target)
    for name in ["val", "test"]:
        for row in read_table(os.path.join(split_dir, "%s.parquet" % name)):
            out.add((row["user_id"], row["item_id"]))
    return out


def _iter_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "__meta__" in obj:
                continue
            yield obj


def _pseudo_aug(path, target, heldout, max_per=60, max_total=120000):
    rows = []
    counts = Counter()
    if not os.path.exists(path):
        return rows
    for row in _iter_rows(path):
        if row.get("dataset") != target:
            continue
        source_uid = row.get("source_user_id", row.get("user_id"))
        if (source_uid, row["item_id"]) in heldout:
            continue
        pid = row.get("pseudo_user_id", source_uid)
        if counts[pid] >= max_per:
            continue
        counts[pid] += 1
        out = dict(row)
        out["user_id"] = "abl:%s" % pid
        rows.append(out)
        if len(rows) >= max_total:
            break
    return rows


def run_ablation(config):
    variants = {
        "single_domain": None,
        "random_mix": "exp_set",
        "v1_pseudo_user": "pseudo_user_interactions.parquet",
        "v2_pseudo_user": "pseudo_user_interactions_v2.parquet",
        "v2fix_pseudo_user": "pseudo_user_interactions_v2fix.parquet",
    }
    results = {}
    lines = []
    k = int(config["experiment"]["k_values"][0])
    model_name = "item_item"
    max_total = int(config["experiment"].get("ablation", {}).get("max_aug_rows_per_variant", 120000))
    for target in config["experiment"]["target_domains"]:
        split_dir = os.path.join(config["paths"]["exp_splits"], target)
        base = read_table(os.path.join(split_dir, "train.parquet"))
        test = read_table(os.path.join(split_dir, "test.parquet"))
        heldout = _heldout(config, target)
        results[target] = {}
        for name, source in variants.items():
            if name == "single_domain":
                train = base
            elif name == "random_mix":
                train = read_table(os.path.join(config["paths"]["exp_sets"], target, "random_mix.parquet"))
            else:
                train = base + _pseudo_aug(os.path.join(config["paths"]["processed"], source), target, heldout, max_total=max_total)
            model = train_model(model_name, train, config)
            metrics = evaluate_model(model, base, test, k)
            metrics["train_rows"] = len(train)
            results[target][name] = metrics
            lines.append("- %s/%s %s Recall@%s=%s NDCG@%s=%s MRR@%s=%s train_rows=%s" % (
                target, name, model_name, k, metrics["recall"], k, metrics["ndcg"], k, metrics["mrr"], metrics["train_rows"]
            ))
        best = max(results[target].items(), key=lambda kv: kv[1]["recall"])[0]
        lines.append("  best_by_recall=%s" % best)
    out = os.path.join(config["paths"]["processed"], "exp_ablation_v3.json")
    write_json(out, results)
    write_markdown("reports/v3_ablation.md", "V3 Ablation", [("Smoke Test", lines)])
    return {"path": out, "results": results, "summary": lines}
