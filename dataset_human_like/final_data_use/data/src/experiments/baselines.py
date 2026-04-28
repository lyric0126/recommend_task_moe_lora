import math
import os
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table


def _build_histories(rows):
    histories = defaultdict(list)
    for row in rows:
        histories[row["user_id"]].append(row["item_id"])
    return histories


class PopularityModel(object):
    def fit(self, rows):
        self.pop = Counter(row["item_id"] for row in rows)
        self.ranked = [item for item, _ in self.pop.most_common()]
        return self

    def recommend(self, history, k=10):
        seen = set(history)
        return [item for item in self.ranked if item not in seen][:k]


class ItemItemModel(object):
    def __init__(self, max_history=80):
        self.max_history = max_history

    def fit(self, rows):
        self.pop = Counter(row["item_id"] for row in rows)
        histories = _build_histories(rows)
        self.co = defaultdict(Counter)
        for items in histories.values():
            uniq = list(dict.fromkeys(items[-self.max_history:]))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:]:
                    self.co[a][b] += 1
                    self.co[b][a] += 1
        self.ranked = [item for item, _ in self.pop.most_common()]
        return self

    def recommend(self, history, k=10):
        seen = set(history)
        scores = Counter()
        for item in list(dict.fromkeys(history[-self.max_history:])):
            for other, val in self.co.get(item, {}).most_common(200):
                if other not in seen:
                    scores[other] += val
        for item, val in self.pop.most_common(300):
            if item not in seen:
                scores[item] += 0.001 * val
        ranked = [item for item, _ in scores.most_common(k)]
        if len(ranked) < k:
            ranked += [item for item in self.ranked if item not in seen and item not in ranked][: k - len(ranked)]
        return ranked[:k]


def train_model(name, rows, config):
    if name == "popularity":
        return PopularityModel().fit(rows)
    if name == "item_item":
        return ItemItemModel(int(config["experiment"]["baseline"].get("item_item_max_history", 80))).fit(rows)
    raise ValueError("unknown model %s" % name)


def evaluate_model(model, train_history_rows, test_rows, k=10):
    histories = _build_histories(train_history_rows)
    metrics = {"recall": 0.0, "ndcg": 0.0, "hitrate": 0.0, "mrr": 0.0, "users": 0}
    for row in test_rows:
        uid = row["user_id"]
        truth = row["item_id"]
        history = histories.get(uid, [])
        if not history:
            continue
        recs = model.recommend(history, k)
        metrics["users"] += 1
        if truth in recs:
            rank = recs.index(truth) + 1
            metrics["recall"] += 1.0
            metrics["hitrate"] += 1.0
            metrics["mrr"] += 1.0 / rank
            metrics["ndcg"] += 1.0 / math.log(rank + 1, 2)
    users = float(metrics["users"] or 1)
    for key in ["recall", "ndcg", "hitrate", "mrr"]:
        metrics[key] = round(metrics[key] / users, 6)
    return metrics


def run_baseline_smoke(config):
    target = config["experiment"]["target_domains"][0]
    train_path = os.path.join(config["paths"]["exp_sets"], target, "single_domain.parquet")
    rows = read_table(train_path)
    sample = rows[:5000]
    model = train_model("item_item", sample, config)
    histories = _build_histories(sample)
    first_uid = sorted(histories.keys())[0]
    recs = model.recommend(histories[first_uid], 10)
    out = "data/processed/exp_baseline_smoke.parquet"
    write_table(out, [{"target": target, "model": "item_item", "user_id": first_uid, "recommendations": recs}], ["target", "model", "user_id", "recommendations"])
    lines = ["target=%s" % target, "sample_train_rows=%s" % len(sample), "user_id=%s recommendations=%s" % (first_uid, recs)]
    write_markdown("reports/v3_baselines.md", "V3 Baselines", [("Smoke Test", lines)])
    return {"path": out, "summary": lines}
