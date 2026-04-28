import hashlib
import math
import os
import re
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_markdown, write_table


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
STOPWORDS = set("""
the a an and or of to in on for with from by at as is are was were be been this that these those
movie film book story news video unknown currently reading to-read favorites ya young adult
""".split())

BROAD_MAP = {
    "comedy": "topic_comedy", "funny": "topic_comedy", "搞笑": "topic_comedy", "喜剧": "topic_comedy",
    "drama": "topic_drama", "romance": "topic_romance", "love": "topic_romance",
    "thriller": "topic_thriller", "mystery": "topic_thriller", "crime": "topic_thriller",
    "fantasy": "topic_fantasy", "magic": "topic_fantasy", "sci": "topic_scifi", "science": "topic_scifi",
    "news": "topic_news", "politics": "topic_news", "社会": "topic_news", "民生": "topic_news",
    "health": "topic_health", "sports": "topic_sports", "music": "topic_music",
    "lifestyle": "topic_lifestyle", "fashion": "topic_lifestyle", "时尚": "topic_lifestyle",
    "travel": "topic_travel", "food": "topic_food", "美食": "topic_food",
}


def _tokens(text):
    raw = [t.lower() for t in TOKEN_RE.findall(text or "")]
    toks = []
    for token in raw:
        if len(token) <= 1 or token in STOPWORDS:
            continue
        toks.append(token)
        for key, broad in BROAD_MAP.items():
            if key in token:
                toks.append(broad)
    return toks[:500]


def _clean(value):
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[_|,;/\\#]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def item_text_v2fix(item):
    ds = item.get("dataset", "")
    category = _clean(item.get("item_category", ""))
    if ds == "movielens":
        title = _clean(item.get("title", item.get("item_text", "")))
        return "title %s genres %s" % (title, category)
    if ds == "goodreads":
        shelves = " ".join(item.get("shelves", []) if isinstance(item.get("shelves"), list) else [])
        authors = " ".join(item.get("authors", []) if isinstance(item.get("authors"), list) else [])
        return "title %s authors %s shelves %s description %s category %s" % (
            _clean(item.get("title", item.get("item_text", ""))), _clean(authors), _clean(shelves), _clean(item.get("item_text", "")), category
        )
    if ds == "mind":
        return "title %s category %s subcategory %s abstract %s" % (
            _clean(item.get("title", item.get("item_text", ""))), _clean(item.get("category", category)), _clean(item.get("subcategory", "")), _clean(item.get("abstract", item.get("item_text", "")))
        )
    if ds == "kuairec":
        return "caption %s category %s tags %s" % (_clean(item.get("caption", item.get("item_text", ""))), category, _clean(item.get("tags", "")))
    return _clean(item.get("item_text", ""))


def _hash(token, dim):
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % dim


def _embed(tokens, df, n_docs, dim):
    tf = Counter(tokens)
    vec = [0.0] * dim
    for token, count in tf.items():
        doc_freq = df.get(token, 1)
        if doc_freq < 2 and not token.startswith("topic_"):
            continue
        idf = math.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        weight = (1.0 + math.log(count)) * idf
        vec[_hash(token, dim)] += weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def build_item_embeddings_v2fix(config):
    dim = int(config["pipeline"].get("embedding_dim", 128))
    docs = []
    for ds in config["datasets"]:
        items = read_table(os.path.join(config["paths"]["interim_clean"], ds, "items.parquet"))
        for item in items:
            text = item_text_v2fix(item)
            docs.append((ds, item, text, _tokens(text)))
    df = Counter()
    for _, _, _, toks in docs:
        df.update(set(toks))
    n_docs = float(len(docs) or 1)
    rows = []
    coverage = defaultdict(int)
    samples = []
    for ds, item, text, toks in docs:
        emb = _embed(toks, df, n_docs, dim)
        row = {
            "dataset": ds,
            "item_id": item["item_id"],
            "item_text_v2fix": text[:2400],
            "item_category": item.get("item_category", "unknown"),
            "embedding_backend": "hash_tfidf_v2fix",
            "embedding_dim": dim,
            "embedding": emb,
            "token_count": len(toks),
            "unique_token_count": len(set(toks)),
        }
        rows.append(row)
        coverage[ds] += 1
        if len(samples) < 8:
            samples.append("- %s/%s text=`%s` emb_head=%s" % (ds, item["item_id"], text[:160], emb[:6]))
    out = os.path.join(config["paths"]["processed"], "item_embeddings_v2fix.parquet")
    write_table(out, rows, ["dataset", "item_id", "item_text_v2fix", "embedding"], config["pipeline"].get("storage_format", "auto"))
    by_ds = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    sim_checks = []
    for ds, ds_rows in by_ds.items():
        if len(ds_rows) >= 2:
            sim_checks.append("%s first_pair_cosine=%s" % (ds, round(cosine(ds_rows[0]["embedding"], ds_rows[1]["embedding"]), 6)))
    old = read_table(os.path.join(config["paths"]["processed"], "item_embeddings_v2.parquet"), limit=3)
    lines = [
        "v2fix_rows=%s" % len(rows),
        "v2fix_dim=%s" % dim,
        "coverage=%s" % dict(coverage),
        "v2_sample_dim=%s" % (len(old[0]["embedding"]) if old else 0),
        "similarity_checks=%s" % sim_checks,
    ] + samples
    write_markdown("reports/v2fix_embeddings.md", "V2fix Embeddings", [("Smoke Test", lines)])
    return {"path": out, "rows": len(rows), "dim": dim, "coverage": dict(coverage), "samples": samples}
