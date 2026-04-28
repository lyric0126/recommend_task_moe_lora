import datetime as dt
import math
import os
from collections import defaultdict

from src.io_utils import read_table, write_markdown, write_table
from src.schema import PROFILE_COLUMNS


def _cosine(a, b):
    den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return sum(x * y for x, y in zip(a, b)) / den


def _mean_vec(vectors):
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            out[i] += float(val)
    return [round(v / len(vectors), 6) for v in out]


def _behavior(dataset, rows):
    vals = [float(r.get("event_value") or 0) for r in rows]
    avg = sum(vals) / len(vals) if vals else 0.0
    if dataset == "movielens":
        return {"rating_strength": round(avg / 5.0, 6)}
    if dataset == "goodreads":
        read_rate = sum(1 for r in rows if r.get("raw_event") == "read") / float(len(rows) or 1)
        return {"interaction_intensity": round((avg / 5.0 + read_rate) / 2.0, 6)}
    if dataset == "mind":
        clicks = sum(1 for r in rows if float(r.get("event_value") or 0) > 0)
        return {"click_impression_rate": round(clicks / float(len(rows) or 1), 6)}
    return {"watch_completion_tendency": round(min(avg, 2.0) / 2.0, 6)}


def _session_stats(timestamps):
    if not timestamps:
        return 0, 0.0
    times = sorted(timestamps)
    sessions = 1
    lens = []
    cur = 1
    last = times[0]
    for ts in times[1:]:
        if ts - last > 1800:
            sessions += 1
            lens.append(cur)
            cur = 1
        else:
            cur += 1
        last = ts
    lens.append(cur)
    return sessions, sum(lens) / float(len(lens))


def build_user_profiles(config):
    emb_rows = read_table(os.path.join(config["paths"]["processed"], "item_embeddings.parquet"))
    emb = {(r["dataset"], r["item_id"]): r["embedding"] for r in emb_rows}
    profiles = []
    samples = []
    for dataset in config["datasets"]:
        interactions = read_table(os.path.join(config["paths"]["interim_clean"], dataset, "interactions.parquet"))
        by_user = defaultdict(list)
        for row in interactions:
            by_user[row["user_id"]].append(row)
        for user_id, rows in by_user.items():
            timestamps = [int(r.get("timestamp") or 0) for r in rows]
            days = {dt.datetime.utcfromtimestamp(ts).date().isoformat() for ts in timestamps if ts > 0}
            hist = [0] * 24
            weekdays = 0
            weekends = 0
            for ts in timestamps:
                if ts <= 0:
                    continue
                d = dt.datetime.utcfromtimestamp(ts)
                hist[d.hour] += 1
                if d.weekday() >= 5:
                    weekends += 1
                else:
                    weekdays += 1
            vectors = [emb[(dataset, r["item_id"])] for r in rows if (dataset, r["item_id"]) in emb]
            session_count, avg_session_len = _session_stats(timestamps)
            profile = {
                "dataset": dataset,
                "user_id": user_id,
                "semantic": _mean_vec(vectors),
                "total_interactions": len(rows),
                "active_days": len(days),
                "session_count": session_count,
                "avg_session_len": round(avg_session_len, 6),
                "hour_hist": hist,
                "weekday_weekend_ratio": round(weekdays / float(weekends or 1), 6),
                "behavior": _behavior(dataset, rows),
            }
            profiles.append(profile)
        if by_user:
            samples.append("- %s profile sample: `%s`" % (dataset, profiles[-1]))
    out = os.path.join(config["paths"]["processed"], "user_profiles.parquet")
    write_table(out, profiles, PROFILE_COLUMNS)
    write_markdown("reports/stage_5_user_profiles_summary.md", "Stage 5 User Profiles Summary", [("Samples", samples), ("Output", out)])
    return {"rows": len(profiles), "path": out, "samples": samples}


def cosine(a, b):
    return _cosine(a, b)
