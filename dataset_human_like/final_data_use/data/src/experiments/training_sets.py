import os
import random
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table


def _heldout_pairs(config, target):
    pairs = set()
    split_dir = os.path.join(config["paths"]["exp_splits"], target)
    for name in ["val", "test"]:
        for row in read_table(os.path.join(split_dir, "%s.parquet" % name)):
            pairs.add((row["user_id"], row["item_id"]))
    return pairs


def _stats(rows):
    return {
        "rows": len(rows),
        "users": len({r["user_id"] for r in rows}),
        "items": len({r["item_id"] for r in rows}),
        "domains": dict(Counter(r.get("dataset", "unknown") for r in rows)),
    }


def _target_pseudo_rows(config, target, pseudo_path, heldout):
    max_per = int(config["experiment"]["training"].get("max_interactions_per_aug_user", 80))
    rows = []
    counts = Counter()
    for row in read_table(pseudo_path):
        if row.get("dataset") != target:
            continue
        source_uid = row.get("source_user_id", row.get("user_id"))
        if (source_uid, row["item_id"]) in heldout:
            continue
        key = row.get("pseudo_user_id", source_uid)
        if counts[key] >= max_per:
            continue
        counts[key] += 1
        out = dict(row)
        out["user_id"] = "pseudo:%s" % key
        out["source_user_id"] = source_uid
        rows.append(out)
    return rows


def _random_aug_rows(config, target, base_train, target_count, heldout):
    rng = random.Random(int(config["experiment"]["training"].get("random_seed", 2027)) + len(target))
    by_user = defaultdict(list)
    for row in base_train:
        by_user[row["user_id"]].append(row)
    users = sorted(by_user.keys())
    if not users:
        return []
    rows = []
    max_per = int(config["experiment"]["training"].get("max_interactions_per_aug_user", 80))
    synthetic_idx = 0
    while len(rows) < target_count and synthetic_idx < target_count * 3:
        uid = rng.choice(users)
        synthetic_idx += 1
        selected = list(by_user[uid])[:max_per]
        if not selected:
            continue
        pseudo_id = "random:%s:%06d" % (target, synthetic_idx)
        for row in selected:
            if (uid, row["item_id"]) in heldout:
                continue
            out = dict(row)
            out["user_id"] = pseudo_id
            out["source_user_id"] = uid
            out["pseudo_user_id"] = pseudo_id
            rows.append(out)
            if len(rows) >= target_count:
                break
    return rows


def build_training_sets(config):
    out_root = config["paths"]["exp_sets"]
    files = []
    report = []
    for target in config["experiment"]["target_domains"]:
        split_dir = os.path.join(config["paths"]["exp_splits"], target)
        base_train = read_table(os.path.join(split_dir, "train.parquet"))
        heldout = _heldout_pairs(config, target)
        pseudo_aug = _target_pseudo_rows(config, target, os.path.join(config["paths"]["processed"], "pseudo_user_interactions_v2fix.parquet"), heldout)
        random_aug = _random_aug_rows(config, target, base_train, len(pseudo_aug), heldout)
        variants = {
            "single_domain": base_train,
            "random_mix": base_train + random_aug,
            "pseudo_user_v2fix": base_train + pseudo_aug,
        }
        for name, rows in variants.items():
            path = os.path.join(out_root, target, "%s.parquet" % name)
            write_table(path, rows, ["dataset", "user_id", "item_id", "timestamp", "event_value"])
            files.append(path)
            stat = _stats(rows)
            report.append("- %s/%s stats=%s" % (target, name, stat))
            report.append("  sample=`%s`" % rows[:1])
    write_markdown("reports/v3_training_sets.md", "V3 Training Sets", [("Smoke Test", report)])
    files.append("reports/v3_training_sets.md")
    return {"files": files, "summary": report}
