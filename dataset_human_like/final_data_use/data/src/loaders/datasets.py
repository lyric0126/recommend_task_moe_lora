import csv
import gzip
import json
import os

from src.io_utils import clean_text, parse_time, write_table
from src.schema import CANONICAL_INTERACTION_COLUMNS, CANONICAL_ITEM_COLUMNS, canonical_interaction, canonical_item


def _limit(config, dataset):
    return int(config.get("pipeline", {}).get("sample_limit", {}).get(dataset, 0) or 0)


def _interim_dir(config, dataset):
    return os.path.join(config["paths"]["interim"], dataset)


def _write_outputs(config, dataset, interactions, items):
    out_dir = _interim_dir(config, dataset)
    ipath = os.path.join(out_dir, "interactions.parquet")
    itpath = os.path.join(out_dir, "items.parquet")
    write_table(ipath, interactions, CANONICAL_INTERACTION_COLUMNS)
    write_table(itpath, items, CANONICAL_ITEM_COLUMNS)
    return ipath, itpath


def load_movielens(config):
    dataset = "movielens"
    base = config["paths"]["raw"][dataset]
    limit = _limit(config, dataset)
    movies_path = os.path.join(base, "movies.csv")
    ratings_path = os.path.join(base, "ratings.csv")
    items_by_id = {}
    with open(movies_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"movieId", "title", "genres"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("MovieLens movies.csv columns differ from expected: %s" % reader.fieldnames)
        for row in reader:
            category = clean_text(row.get("genres", "")).replace("|", ";")
            text = clean_text("%s %s" % (row.get("title", ""), category))
            items_by_id[str(row["movieId"])] = canonical_item(dataset, row["movieId"], text, category, title=row.get("title", ""))
    interactions = []
    used_items = set()
    with open(ratings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"userId", "movieId", "rating", "timestamp"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("MovieLens ratings.csv columns differ from expected: %s" % reader.fieldnames)
        for row in reader:
            item = items_by_id.get(str(row["movieId"]), canonical_item(dataset, row["movieId"], "", "unknown"))
            interactions.append(canonical_interaction(dataset, row["userId"], row["movieId"], row["timestamp"], "rating", row["rating"], item["item_text"], item["item_category"]))
            used_items.add(str(row["movieId"]))
            if limit and len(interactions) >= limit:
                break
    items = [items_by_id[i] for i in sorted(used_items) if i in items_by_id]
    paths = _write_outputs(config, dataset, interactions, items)
    return {"dataset": dataset, "interactions": len(interactions), "items": len(items), "paths": paths, "sample": interactions[:3], "columns": CANONICAL_INTERACTION_COLUMNS}


def load_goodreads(config):
    dataset = "goodreads"
    base = config["paths"]["raw"][dataset]
    limit = _limit(config, dataset)
    inter_path = os.path.join(base, "goodreads_interactions_young_adult.json.gz")
    book_path = os.path.join(base, "goodreads_books_young_adult.json.gz")
    interactions = []
    used_items = set()
    with gzip.open(inter_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            expected = {"user_id", "book_id", "rating", "is_read", "date_added"}
            if not expected.issubset(row.keys()):
                raise ValueError("Goodreads interaction columns differ from expected")
            event = "read" if row.get("is_read") else "shelf"
            value = float(row.get("rating") or (1.0 if row.get("is_read") else 0.2))
            interactions.append(canonical_interaction(dataset, row["user_id"], row["book_id"], parse_time(row.get("date_added")), event, value))
            used_items.add(str(row["book_id"]))
            if limit and len(interactions) >= limit:
                break
    items = []
    found = set()
    with gzip.open(book_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            book_id = str(row.get("book_id", ""))
            if book_id not in used_items:
                continue
            shelves = [s.get("name", "") for s in row.get("popular_shelves", [])[:8] if isinstance(s, dict)]
            authors = [a.get("author_id", "") for a in row.get("authors", []) if isinstance(a, dict)]
            category = ";".join(shelves[:5]) or "unknown"
            text = clean_text("%s %s %s %s" % (row.get("title", ""), " ".join(authors), " ".join(shelves), row.get("description", "")))
            items.append(canonical_item(dataset, book_id, text, category, title=row.get("title", ""), authors=authors, shelves=shelves))
            found.add(book_id)
            if len(found) >= len(used_items):
                break
    fallback_items = used_items - found
    for book_id in list(fallback_items)[:1000]:
        items.append(canonical_item(dataset, book_id, "goodreads book %s" % book_id, "unknown"))
    item_map = {r["item_id"]: r for r in items}
    for row in interactions:
        item = item_map.get(row["item_id"])
        if item:
            row["item_text"] = item["item_text"]
            row["item_category"] = item["item_category"]
    paths = _write_outputs(config, dataset, interactions, items)
    return {"dataset": dataset, "interactions": len(interactions), "items": len(items), "paths": paths, "sample": interactions[:3], "columns": CANONICAL_INTERACTION_COLUMNS}


def load_mind(config):
    dataset = "mind"
    base = config["paths"]["raw"][dataset]
    limit = _limit(config, dataset)
    news_path = os.path.join(base, "news.tsv")
    behaviors_path = os.path.join(base, "behaviors.tsv")
    news = {}
    with open(news_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                raise ValueError("MIND news.tsv has fewer columns than expected")
            item_id, cat, subcat, title, abstract = parts[:5]
            text = clean_text("%s %s %s %s" % (title, cat, subcat, abstract))
            news[item_id] = canonical_item(dataset, item_id, text, "%s;%s" % (cat, subcat), title=title, category=cat, subcategory=subcat, abstract=abstract)
    interactions = []
    used_items = set()
    with open(behaviors_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                raise ValueError("MIND behaviors.tsv has fewer columns than expected")
            _, user_id, ts, history, impressions = parts[:5]
            timestamp = parse_time(ts)
            for item_id in history.split():
                item = news.get(item_id, canonical_item(dataset, item_id, "", "unknown"))
                interactions.append(canonical_interaction(dataset, user_id, item_id, timestamp, "history", 0.5, item["item_text"], item["item_category"]))
                used_items.add(item_id)
                if limit and len(interactions) >= limit:
                    break
            if limit and len(interactions) >= limit:
                break
            for imp in impressions.split():
                if "-" not in imp:
                    continue
                item_id, clicked = imp.rsplit("-", 1)
                item = news.get(item_id, canonical_item(dataset, item_id, "", "unknown"))
                interactions.append(canonical_interaction(dataset, user_id, item_id, timestamp, "impression", float(clicked), item["item_text"], item["item_category"]))
                used_items.add(item_id)
                if limit and len(interactions) >= limit:
                    break
            if limit and len(interactions) >= limit:
                break
    items = [news[i] for i in sorted(used_items) if i in news]
    paths = _write_outputs(config, dataset, interactions, items)
    return {"dataset": dataset, "interactions": len(interactions), "items": len(items), "paths": paths, "sample": interactions[:3], "columns": CANONICAL_INTERACTION_COLUMNS}


def load_kuairec(config):
    dataset = "kuairec"
    base = config["paths"]["raw"][dataset]
    limit = _limit(config, dataset)
    matrix_path = os.path.join(base, "big_matrix.csv")
    if not os.path.exists(matrix_path):
        matrix_path = os.path.join(base, "small_matrix.csv")
    item_path = os.path.join(base, "kuairec_caption_category.csv")
    items_by_id = {}
    with open(item_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"video_id", "caption", "topic_tag", "first_level_category_name"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("KuaiRec item columns differ from expected: %s" % reader.fieldnames)
        for row in reader:
            category = clean_text("%s;%s;%s" % (row.get("first_level_category_name", ""), row.get("second_level_category_name", ""), row.get("third_level_category_name", "")))
            text = clean_text("%s %s %s" % (row.get("caption", ""), row.get("manual_cover_text", ""), row.get("topic_tag", "")))
            items_by_id[str(row["video_id"])] = canonical_item(dataset, row["video_id"], text, category, caption=row.get("caption", ""), tags=row.get("topic_tag", ""))
    interactions = []
    used_items = set()
    with open(matrix_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"user_id", "video_id", "timestamp", "watch_ratio"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("KuaiRec matrix columns differ from expected: %s" % reader.fieldnames)
        for row in reader:
            item = items_by_id.get(str(row["video_id"]), canonical_item(dataset, row["video_id"], "", "unknown"))
            interactions.append(canonical_interaction(dataset, row["user_id"], row["video_id"], row["timestamp"], "watch", row.get("watch_ratio", 0), item["item_text"], item["item_category"]))
            used_items.add(str(row["video_id"]))
            if limit and len(interactions) >= limit:
                break
    items = [items_by_id[i] for i in sorted(used_items) if i in items_by_id]
    paths = _write_outputs(config, dataset, interactions, items)
    return {"dataset": dataset, "interactions": len(interactions), "items": len(items), "paths": paths, "sample": interactions[:3], "columns": CANONICAL_INTERACTION_COLUMNS}


LOADERS = {
    "movielens": load_movielens,
    "goodreads": load_goodreads,
    "mind": load_mind,
    "kuairec": load_kuairec,
}


def run_loaders(config):
    return [LOADERS[name](config) for name in config["datasets"]]
