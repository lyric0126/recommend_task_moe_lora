import os
from collections import defaultdict

from src.io_utils import read_table, write_markdown, write_table


TARGETS = ["movielens", "goodreads"]


def _split_rows(rows, min_interactions):
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    train, val, test = [], [], []
    for uid, user_rows in by_user.items():
        ordered = sorted(user_rows, key=lambda r: (int(r.get("timestamp") or 0), r["item_id"]))
        if len(ordered) < min_interactions:
            continue
        train.extend(ordered[:-2])
        val.append(ordered[-2])
        test.append(ordered[-1])
    return train, val, test


def build_splits(config):
    out_root = config["paths"]["exp_splits"]
    min_interactions = int(config["experiment"]["split"].get("min_user_interactions", 5))
    files = []
    lines = []
    for target in config["experiment"]["target_domains"]:
        rows = read_table(os.path.join(config["paths"]["interim_clean"], target, "interactions.parquet"))
        train, val, test = _split_rows(rows, min_interactions)
        target_dir = os.path.join(out_root, target)
        paths = {
            "train": os.path.join(target_dir, "train.parquet"),
            "val": os.path.join(target_dir, "val.parquet"),
            "test": os.path.join(target_dir, "test.parquet"),
        }
        write_table(paths["train"], train, ["dataset", "user_id", "item_id", "timestamp", "event_value"])
        write_table(paths["val"], val, ["dataset", "user_id", "item_id", "timestamp", "event_value"])
        write_table(paths["test"], test, ["dataset", "user_id", "item_id", "timestamp", "event_value"])
        files.extend(paths.values())
        lines.append("- %s train=%s val=%s test=%s users=%s" % (target, len(train), len(val), len(test), len({r["user_id"] for r in train})))
        lines.append("  train_sample=`%s`" % train[:1])
        lines.append("  test_sample=`%s`" % test[:1])
    write_markdown("reports/v3_data_split.md", "V3 Data Split", [("Smoke Test", lines)])
    files.append("reports/v3_data_split.md")
    return {"files": files, "summary": lines}
