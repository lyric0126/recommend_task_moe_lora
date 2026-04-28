import hashlib
import math
import os
import re
from collections import Counter, defaultdict

from src.io_utils import read_table, table_storage_info, write_markdown, write_table


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def enhanced_item_text(item):
    dataset = item.get("dataset", "")
    parts = [item.get("item_text", ""), item.get("item_category", "")]
    if dataset == "goodreads":
        parts.extend([
            item.get("title", ""),
            " ".join(item.get("authors", []) if isinstance(item.get("authors"), list) else []),
            " ".join(item.get("shelves", []) if isinstance(item.get("shelves"), list) else []),
        ])
    elif dataset == "mind":
        parts.extend([item.get("title", ""), item.get("category", ""), item.get("subcategory", ""), item.get("abstract", "")])
    elif dataset == "kuairec":
        parts.extend([item.get("caption", ""), item.get("tags", "")])
    return " ".join(str(p or "") for p in parts)


def _backend_name(config):
    requested = config["pipeline"].get("embedding_backend", "auto")
    if requested not in ("auto", "hash_tfidf"):
        return "hash_tfidf_fallback"
    return "hash_tfidf_fallback"


def _hash_index(token, dim):
    h = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(h[:10], 16) % dim, (1.0 if int(h[10:12], 16) % 2 == 0 else -1.0)


def build_item_embeddings_v2(config):
    dim = int(config["pipeline"].get("embedding_dim", 96))
    docs = []
    for dataset in config["datasets"]:
        for item in read_table(os.path.join(config["paths"]["interim_clean"], dataset, "items.parquet")):
            text = enhanced_item_text(item)
            docs.append((dataset, item, text, _tokens(text)))
    df = Counter()
    for _, _, _, toks in docs:
        df.update(set(toks))
    n_docs = float(len(docs) or 1)
    rows = []
    coverage = defaultdict(int)
    text_samples = []
    backend = _backend_name(config)
    for dataset, item, text, toks in docs:
        tf = Counter(toks)
        vec = [0.0] * dim
        for token, count in tf.items():
            idx, sign = _hash_index(token, dim)
            idf = math.log((1.0 + n_docs) / (1.0 + df[token])) + 1.0
            vec[idx] += sign * (1.0 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        emb = [round(v / norm, 6) for v in vec]
        rows.append({
            "dataset": dataset,
            "item_id": item["item_id"],
            "item_text_v2": text[:2000],
            "item_category": item.get("item_category", "unknown"),
            "embedding_backend": backend,
            "embedding_dim": dim,
            "embedding": emb,
            "token_count": len(toks),
            "unique_token_count": len(set(toks)),
        })
        coverage[dataset] += 1
        if len(text_samples) < 8:
            text_samples.append("- %s/%s text=`%s` emb_head=%s" % (dataset, item["item_id"], text[:180], emb[:6]))
    out = os.path.join(config["paths"]["processed"], "item_embeddings_v2.parquet")
    write_table(out, rows, ["dataset", "item_id", "item_text_v2", "embedding"], config["pipeline"].get("storage_format", "auto"))
    old_rows = len(read_table(os.path.join(config["paths"]["processed"], "item_embeddings.parquet")))
    info = table_storage_info(out)
    lines = [
        "backend=%s" % backend,
        "old_embedding_rows=%s" % old_rows,
        "new_embedding_rows=%s" % len(rows),
        "new_embedding_dim=%s" % dim,
        "coverage=%s" % dict(coverage),
        "storage=%s" % info,
    ] + text_samples
    write_markdown("reports/v2_item_representation.md", "V2 Item Representation", [("Smoke Test", lines)])
    return {"path": out, "rows": len(rows), "dim": dim, "coverage": dict(coverage), "backend": backend, "samples": text_samples}
