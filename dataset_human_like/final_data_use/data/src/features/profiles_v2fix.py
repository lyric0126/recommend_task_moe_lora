import datetime as dt
import math
import os
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table


def _mean(vectors, weights=None):
    if not vectors:
        return []
    dim = len(vectors[0])
    weights = weights or [1.0] * len(vectors)
    total = sum(weights) or 1.0
    out = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i, val in enumerate(vec):
            out[i] += float(val) * w
    return [round(v / total, 6) for v in out]


def _entropy(counter):
    total = float(sum(counter.values()) or 1)
    out = 0.0
    for count in counter.values():
        p = count / total
        if p:
            out -= p * math.log(p, 2)
    return round(out, 6)


def _behavior(ds, rows):
    vals = [float(r.get("event_value") or 0) for r in rows]
    avg = sum(vals) / float(len(vals) or 1)
    b = {"avg_event_value": round(avg, 6)}
    if ds == "movielens":
        b.update({"rating_strength": round(avg / 5.0, 6), "high_rating_share": round(sum(v >= 4 for v in vals) / float(len(vals) or 1), 6)})
    elif ds == "goodreads":
        read = sum(1 for r in rows if r.get("raw_event") == "read") / float(len(rows) or 1)
        b.update({"read_share": round(read, 6), "interaction_intensity": round((read + min(avg / 5.0, 1.0)) / 2.0, 6)})
    elif ds == "mind":
        b.update({"click_rate": round(sum(v > 0 for v in vals) / float(len(vals) or 1), 6)})
    else:
        b.update({"watch_completion_tendency": round(min(avg, 2.0) / 2.0, 6), "strong_watch_share": round(sum(v >= 0.8 for v in vals) / float(len(vals) or 1), 6)})
    return b


def _session(timestamps):
    times = sorted(t for t in timestamps if t > 0)
    if not times:
        return 0, 0.0, 0
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
    gaps = sorted(gaps)
    return len(sessions), round(sum(len(s) for s in sessions) / float(len(sessions)), 6), (gaps[len(gaps) // 2] if gaps else 0)


def build_user_profiles_v2fix(config):
    emb_rows = read_table(os.path.join(config["paths"]["processed"], "item_embeddings_v2fix.parquet"))
    emb = {(r["dataset"], r["item_id"]): r["embedding"] for r in emb_rows}
    profiles = []
    samples = []
    for ds in config["datasets"]:
        rows = read_table(os.path.join(config["paths"]["interim_clean"], ds, "interactions.parquet"))
        by_user = defaultdict(list)
        for row in rows:
            by_user[row["user_id"]].append(row)
        for uid, user_rows in by_user.items():
            user_rows = sorted(user_rows, key=lambda r: int(r.get("timestamp") or 0))
            timestamps = [int(r.get("timestamp") or 0) for r in user_rows]
            max_ts = max(timestamps) if timestamps else 0
            vectors, recent_weights = [], []
            cats = Counter()
            hour = [0] * 24
            dow = [0] * 7
            for r in user_rows:
                vec = emb.get((ds, r["item_id"]))
                if vec:
                    weight = max(0.15, float(r.get("event_value") or 0.2))
                    vectors.append([x * weight for x in vec])
                    age_days = max(0.0, (max_ts - int(r.get("timestamp") or 0)) / 86400.0)
                    recent_weights.append(math.exp(-age_days / 60.0))
                cats.update([c for c in str(r.get("item_category", "unknown")).split(";") if c])
                ts = int(r.get("timestamp") or 0)
                if ts > 0:
                    d = dt.datetime.utcfromtimestamp(ts)
                    hour[d.hour] += 1
                    dow[d.weekday()] += 1
            session_count, avg_session_len, median_gap = _session(timestamps)
            weekdays, weekends = sum(dow[:5]), sum(dow[5:])
            profile = {
                "dataset": ds,
                "user_id": uid,
                "semantic_long_term": _mean(vectors),
                "semantic_recent": _mean(vectors, recent_weights),
                "topic_entropy": _entropy(cats),
                "topic_count": len(cats),
                "top_topics": [k for k, _ in cats.most_common(8)],
                "activity_total_interactions": len(user_rows),
                "activity_active_days": len({dt.datetime.utcfromtimestamp(ts).date().isoformat() for ts in timestamps if ts > 0}),
                "activity_session_count": session_count,
                "activity_avg_session_len": avg_session_len,
                "activity_median_session_gap_sec": median_gap,
                "temporal_hour_hist": hour,
                "temporal_dow_hist": dow,
                "temporal_weekday_weekend_ratio": round(weekdays / float(weekends or 1), 6),
                "temporal_peak_hour": max(range(24), key=lambda i: hour[i]),
                "behavior_summary": _behavior(ds, user_rows),
            }
            profiles.append(profile)
        one = [p for p in profiles if p["dataset"] == ds][:1]
        samples.append("- %s sample=`%s`" % (ds, one))
    out = os.path.join(config["paths"]["processed"], "user_profiles_v2fix.parquet")
    write_table(out, profiles, ["dataset", "user_id", "semantic_long_term", "semantic_recent", "activity_*", "temporal_*"], config["pipeline"].get("storage_format", "auto"))
    v2 = read_table(os.path.join(config["paths"]["processed"], "user_profiles_v2.parquet"), limit=1)
    lines = ["v2_rows_available=%s" % bool(v2), "v2fix_rows=%s" % len(profiles), "field_count=%s" % (len(profiles[0]) if profiles else 0)] + samples
    write_markdown("reports/v2fix_user_profiles.md", "V2fix User Profiles", [("Smoke Test", lines)])
    return {"path": out, "rows": len(profiles), "samples": samples}
