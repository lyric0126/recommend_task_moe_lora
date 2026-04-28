import math
import os
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_markdown, write_table


def _activity_bucket(profile):
    total = int(profile.get("total_interactions") or 0)
    if total < 30:
        return "low"
    if total < 100:
        return "mid"
    return "high"


def _temporal_bucket(profile):
    hist = profile.get("hour_hist") or [0] * 24
    peak = max(range(len(hist)), key=lambda i: hist[i]) if hist else 0
    if peak < 8:
        return "night"
    if peak < 16:
        return "day"
    return "evening"


def block_key(profile):
    return _activity_bucket(profile) + "|" + _temporal_bucket(profile)


def _activity_score(a, b):
    fields = ["total_interactions", "active_days", "session_count", "avg_session_len"]
    vals = []
    for field in fields:
        x = float(a.get(field) or 0)
        y = float(b.get(field) or 0)
        vals.append(1.0 - abs(x - y) / max(x, y, 1.0))
    return sum(vals) / len(vals)


def _temporal_score(a, b):
    return max(0.0, cosine(a.get("hour_hist") or [], b.get("hour_hist") or []))


def _behavior_scalar(profile):
    behavior = profile.get("behavior") or {}
    if not behavior:
        return 0.0
    return float(list(behavior.values())[0] or 0.0)


def _behavior_score(a, b):
    x = _behavior_scalar(a)
    y = _behavior_scalar(b)
    return 1.0 - abs(x - y) / max(x, y, 1.0)


def pair_score(a, b):
    semantic = max(0.0, cosine(a.get("semantic") or [], b.get("semantic") or []))
    activity = _activity_score(a, b)
    temporal = _temporal_score(a, b)
    behavior = _behavior_score(a, b)
    score = 0.5 * semantic + 0.2 * activity + 0.2 * temporal + 0.1 * behavior
    return round(score, 6), {
        "semantic": round(semantic, 6),
        "activity": round(activity, 6),
        "temporal": round(temporal, 6),
        "behavior": round(behavior, 6),
    }


def top_matches(left_profiles, right_profiles, top_k=5):
    right_blocks = defaultdict(list)
    for prof in right_profiles:
        right_blocks[block_key(prof)].append(prof)
    all_right = right_profiles[:1000]
    out = []
    candidate_counts = []
    for left in left_profiles:
        candidates = right_blocks.get(block_key(left)) or all_right
        candidate_counts.append(len(candidates))
        scored = []
        for right in candidates:
            score, parts = pair_score(left, right)
            scored.append((score, right, parts))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, right, parts) in enumerate(scored[:top_k], 1):
            out.append({
                "left_dataset": left["dataset"],
                "left_user_id": left["user_id"],
                "right_dataset": right["dataset"],
                "right_user_id": right["user_id"],
                "rank": rank,
                "score": score,
                "score_parts": parts,
                "block_key": block_key(left),
                "candidate_count": len(candidates),
            })
    return out, candidate_counts


def run_pair(config, left_dataset, right_dataset):
    profiles = read_table(os.path.join(config["paths"]["processed"], "user_profiles.parquet"))
    left = [p for p in profiles if p["dataset"] == left_dataset]
    right = [p for p in profiles if p["dataset"] == right_dataset]
    matches, candidate_counts = top_matches(left, right, int(config["pipeline"].get("top_k_matches", 5)))
    out = os.path.join(config["paths"]["processed"], "matches_%s_%s.parquet" % (left_dataset, right_dataset))
    write_table(out, matches, ["left_dataset", "left_user_id", "right_dataset", "right_user_id", "score"])
    return {"pair": "%s_%s" % (left_dataset, right_dataset), "rows": len(matches), "path": out, "sample": matches[:5], "candidate_counts": candidate_counts}


def run_matching(config):
    results = [
        run_pair(config, "movielens", "goodreads"),
        run_pair(config, "mind", "kuairec"),
    ]
    lines = []
    for r in results:
        counts = Counter(r["candidate_counts"])
        lines.append("- %s rows=%s output=%s" % (r["pair"], r["rows"], r["path"]))
        lines.append("  candidate_count_distribution=%s" % dict(counts.most_common(8)))
        lines.append("  sample=`%s`" % (r["sample"][:2],))
    write_markdown("reports/stage_6_matching_summary.md", "Stage 6 Matching Summary", [("Results", lines)])
    return results
