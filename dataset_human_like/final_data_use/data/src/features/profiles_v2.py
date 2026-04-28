import datetime as dt
import math
import os
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_markdown, write_table


def _mean_vec(vectors, weights=None):
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    weights = weights or [1.0] * len(vectors)
    total = sum(weights) or 1.0
    for vec, weight in zip(vectors, weights):
        for i, val in enumerate(vec):
            out[i] += float(val) * weight
    return [round(v / total, 6) for v in out]


def _entropy(counts):
    total = float(sum(counts.values()) or 1)
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return round(ent, 6)


def _session_stats(timestamps):
    times = sorted(t for t in timestamps if t > 0)
    if not times:
        return {"session_count": 0, "avg_session_len": 0.0, "median_session_gap_sec": 0.0}
    sessions = []
    cur = [times[0]]
    gaps = []
    for prev, ts in zip(times, times[1:]):
        gap = ts - prev
        gaps.append(gap)
        if gap > 1800:
            sessions.append(cur)
            cur = [ts]
        else:
            cur.append(ts)
    sessions.append(cur)
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else 0
    return {
        "session_count": len(sessions),
        "avg_session_len": round(sum(len(s) for s in sessions) / float(len(sessions)), 6),
        "median_session_gap_sec": median_gap,
    }


def _behavior(dataset, rows):
    vals = [float(r.get("event_value") or 0) for r in rows]
    avg = sum(vals) / float(len(vals) or 1)
    summary = {"avg_event_value": round(avg, 6), "max_event_value": round(max(vals) if vals else 0.0, 6)}
    if dataset == "movielens":
        summary["rating_strength"] = round(avg / 5.0, 6)
        summary["high_rating_share"] = round(sum(1 for v in vals if v >= 4.0) / float(len(vals) or 1), 6)
    elif dataset == "goodreads":
        summary["read_share"] = round(sum(1 for r in rows if r.get("raw_event") == "read") / float(len(rows) or 1), 6)
        summary["interaction_intensity"] = round((summary["read_share"] + min(avg / 5.0, 1.0)) / 2.0, 6)
    elif dataset == "mind":
        summary["click_rate"] = round(sum(1 for v in vals if v > 0) / float(len(vals) or 1), 6)
        summary["history_share"] = round(sum(1 for r in rows if r.get("raw_event") == "history") / float(len(rows) or 1), 6)
    else:
        summary["watch_completion_tendency"] = round(min(avg, 2.0) / 2.0, 6)
        summary["strong_watch_share"] = round(sum(1 for v in vals if v >= 0.8) / float(len(vals) or 1), 6)
    return summary


def build_user_profiles_v2(config):
    emb_rows = read_table(os.path.join(config["paths"]["processed"], "item_embeddings_v2.parquet"))
    emb = {(r["dataset"], r["item_id"]): r["embedding"] for r in emb_rows}
    profiles = []
    samples = []
    for dataset in config["datasets"]:
        interactions = read_table(os.path.join(config["paths"]["interim_clean"], dataset, "interactions.parquet"))
        by_user = defaultdict(list)
        for row in interactions:
            by_user[row["user_id"]].append(row)
        for user_id, rows in by_user.items():
            rows = sorted(rows, key=lambda r: int(r.get("timestamp") or 0))
            timestamps = [int(r.get("timestamp") or 0) for r in rows]
            max_ts = max(timestamps) if timestamps else 0
            vectors, recency_weights = [], []
            category_counts = Counter()
            hour_hist = [0] * 24
            dow_hist = [0] * 7
            for r in rows:
                key = (dataset, r["item_id"])
                if key in emb:
                    vectors.append(emb[key])
                    age_days = max(0.0, (max_ts - int(r.get("timestamp") or 0)) / 86400.0)
                    recency_weights.append(math.exp(-age_days / 90.0))
                category_counts.update([c for c in str(r.get("item_category", "unknown")).split(";") if c])
                ts = int(r.get("timestamp") or 0)
                if ts > 0:
                    d = dt.datetime.utcfromtimestamp(ts)
                    hour_hist[d.hour] += 1
                    dow_hist[d.weekday()] += 1
            weekdays = sum(dow_hist[:5])
            weekends = sum(dow_hist[5:])
            session = _session_stats(timestamps)
            activity = {
                "total_interactions": len(rows),
                "active_days": len({dt.datetime.utcfromtimestamp(ts).date().isoformat() for ts in timestamps if ts > 0}),
                "session_count": session["session_count"],
                "avg_session_len": session["avg_session_len"],
                "median_session_gap_sec": session["median_session_gap_sec"],
            }
            temporal = {
                "hour_hist": hour_hist,
                "dow_hist": dow_hist,
                "weekday_weekend_ratio": round(weekdays / float(weekends or 1), 6),
                "peak_hour": max(range(24), key=lambda i: hour_hist[i]) if hour_hist else 0,
            }
            profile = {
                "dataset": dataset,
                "user_id": user_id,
                "semantic_mean": _mean_vec(vectors),
                "semantic_recency": _mean_vec(vectors, recency_weights),
                "topic_entropy": _entropy(category_counts),
                "topic_count": len(category_counts),
                "top_topics": [k for k, _ in category_counts.most_common(8)],
                "activity_total_interactions": activity["total_interactions"],
                "activity_active_days": activity["active_days"],
                "activity_session_count": activity["session_count"],
                "activity_avg_session_len": activity["avg_session_len"],
                "activity_median_session_gap_sec": activity["median_session_gap_sec"],
                "temporal_hour_hist": temporal["hour_hist"],
                "temporal_dow_hist": temporal["dow_hist"],
                "temporal_weekday_weekend_ratio": temporal["weekday_weekend_ratio"],
                "temporal_peak_hour": temporal["peak_hour"],
                "behavior_summary": _behavior(dataset, rows),
            }
            profiles.append(profile)
        if by_user:
            sample = [p for p in profiles if p["dataset"] == dataset][:1]
            samples.append("- %s sample: `%s`" % (dataset, sample))
    out = os.path.join(config["paths"]["processed"], "user_profiles_v2.parquet")
    write_table(out, profiles, ["dataset", "user_id", "semantic_mean", "semantic_recency", "activity_*", "temporal_*", "behavior_summary"], config["pipeline"].get("storage_format", "auto"))
    old = read_table(os.path.join(config["paths"]["processed"], "user_profiles.parquet"), limit=1)
    lines = [
        "v1_field_count=%s" % (len(old[0].keys()) if old else 0),
        "v2_field_count=%s" % (len(profiles[0].keys()) if profiles else 0),
        "v2_rows=%s" % len(profiles),
    ] + samples
    write_markdown("reports/v2_user_profiles.md", "V2 User Profiles", [("Smoke Test", lines)])
    return {"path": out, "rows": len(profiles), "samples": samples}
