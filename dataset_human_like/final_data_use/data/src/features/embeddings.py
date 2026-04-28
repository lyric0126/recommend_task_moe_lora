import hashlib
import math
import os
import re

from src.io_utils import read_table, write_markdown, write_table


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def hash_embedding(text, dim):
    vec = [0.0] * dim
    for token in tokenize(text):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def build_item_embeddings(config):
    dim = int(config["pipeline"]["embedding_dim"])
    rows = []
    samples = []
    for dataset in config["datasets"]:
        items = read_table(os.path.join(config["paths"]["interim_clean"], dataset, "items.parquet"))
        for item in items:
            text = "%s %s" % (item.get("item_text", ""), item.get("item_category", ""))
            row = {
                "dataset": dataset,
                "item_id": item["item_id"],
                "item_text": item.get("item_text", ""),
                "item_category": item.get("item_category", "unknown"),
                "embedding_model": "deterministic_hashing_fallback",
                "embedding": hash_embedding(text, dim),
            }
            rows.append(row)
        if items:
            samples.append("- %s sample item=%s dim=%s" % (dataset, items[0]["item_id"], dim))
    out = os.path.join(config["paths"]["processed"], "item_embeddings.parquet")
    write_table(out, rows, ["dataset", "item_id", "embedding"])
    write_markdown("reports/stage_4_embedding_summary.md", "Stage 4 Embedding Summary", [("Fallback", "Used deterministic hashing fallback."), ("Samples", samples), ("Output", out)])
    return {"rows": len(rows), "dim": dim, "path": out, "samples": samples}
