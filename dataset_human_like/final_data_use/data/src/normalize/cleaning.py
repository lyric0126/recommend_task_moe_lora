import os
from collections import Counter

from src.io_utils import clean_text, read_table, write_markdown, write_table
from src.schema import CANONICAL_INTERACTION_COLUMNS, CANONICAL_ITEM_COLUMNS


def _standard_category(value):
    text = clean_text(value).lower()
    return text.replace("|", ";").replace(",", ";") or "unknown"


def clean_dataset(config, dataset):
    min_u = int(config["pipeline"]["min_interactions_per_user"])
    min_i = int(config["pipeline"]["min_interactions_per_item"])
    src_dir = os.path.join(config["paths"]["interim"], dataset)
    dst_dir = os.path.join(config["paths"]["interim_clean"], dataset)
    interactions = read_table(os.path.join(src_dir, "interactions.parquet"))
    items = read_table(os.path.join(src_dir, "items.parquet"))
    before = len(interactions)
    user_counts = Counter(r["user_id"] for r in interactions)
    item_counts = Counter(r["item_id"] for r in interactions)
    kept = []
    for row in interactions:
        if user_counts[row["user_id"]] >= min_u and item_counts[row["item_id"]] >= min_i:
            row["timestamp"] = int(float(row.get("timestamp") or 0))
            row["item_text"] = clean_text(row.get("item_text", ""))
            row["item_category"] = _standard_category(row.get("item_category", ""))
            kept.append(row)
    kept_items = {r["item_id"] for r in kept}
    clean_items = []
    for item in items:
        if item["item_id"] in kept_items:
            item["item_text"] = clean_text(item.get("item_text", ""))
            item["item_category"] = _standard_category(item.get("item_category", ""))
            clean_items.append(item)
    ipath = os.path.join(dst_dir, "interactions.parquet")
    itpath = os.path.join(dst_dir, "items.parquet")
    write_table(ipath, kept, CANONICAL_INTERACTION_COLUMNS)
    write_table(itpath, clean_items, CANONICAL_ITEM_COLUMNS)
    return {
        "dataset": dataset,
        "before_interactions": before,
        "after_interactions": len(kept),
        "before_items": len(items),
        "after_items": len(clean_items),
        "interaction_sample": kept[:1],
        "item_sample": clean_items[:1],
        "paths": [ipath, itpath],
    }


def run_cleaning(config):
    results = [clean_dataset(config, d) for d in config["datasets"]]
    lines = []
    for r in results:
        lines.append("- %s: interactions %s -> %s; items %s -> %s" % (r["dataset"], r["before_interactions"], r["after_interactions"], r["before_items"], r["after_items"]))
        lines.append("  interaction sample: `%s`" % (r["interaction_sample"][:1],))
        lines.append("  item sample: `%s`" % (r["item_sample"][:1],))
    write_markdown("reports/stage_3_cleaning_summary.md", "Stage 3 Cleaning Summary", [("Results", lines)])
    return results
