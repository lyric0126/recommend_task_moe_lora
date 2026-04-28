import os
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_markdown, write_table


def _weights(config):
    return config["pipeline"].get("matching_weights", {})


def _thresholds(config):
    return config["pipeline"].get("confidence_thresholds", {"strict": 0.7, "medium": 0.52})


def confidence(score, config):
    th = _thresholds(config)
    if score >= float(th.get("strict", 0.7)):
        return "strict"
    if score >= float(th.get("medium", 0.52)):
        return "medium"
    return "loose"


def _activity_bucket(p):
    total = int(p.get("activity_total_interactions") or 0)
    if total < 30:
        return "low"
    if total < 100:
        return "mid"
    return "high"


def _temporal_bucket(p):
    peak = int(p.get("temporal_peak_hour") or 0)
    if peak < 8:
        return "night"
    if peak < 16:
        return "day"
    return "evening"


def _semantic_bucket(p):
    vec = p.get("semantic_mean") or []
    if not vec:
        return "sem0"
    idx = max(range(len(vec)), key=lambda i: abs(vec[i]))
    sign = "p" if vec[idx] >= 0 else "n"
    return "sem%s%s" % (idx % 12, sign)


def block_key(p, config):
    parts = [_activity_bucket(p), _temporal_bucket(p)]
    if config["pipeline"].get("semantic_coarse_block", True):
        parts.append(_semantic_bucket(p))
    return "|".join(parts)


def _activity_score(a, b):
    fields = [
        "activity_total_interactions",
        "activity_active_days",
        "activity_session_count",
        "activity_avg_session_len",
    ]
    vals = []
    for field in fields:
        x = float(a.get(field) or 0)
        y = float(b.get(field) or 0)
        vals.append(1.0 - abs(x - y) / max(x, y, 1.0))
    return sum(vals) / len(vals)


def _temporal_score(a, b):
    hour = max(0.0, cosine(a.get("temporal_hour_hist") or [], b.get("temporal_hour_hist") or []))
    dow = max(0.0, cosine(a.get("temporal_dow_hist") or [], b.get("temporal_dow_hist") or []))
    return 0.75 * hour + 0.25 * dow


def _behavior_scalar(p):
    behavior = p.get("behavior_summary") or {}
    preferred = [
        "rating_strength",
        "interaction_intensity",
        "click_rate",
        "watch_completion_tendency",
        "avg_event_value",
    ]
    for key in preferred:
        if key in behavior:
            return float(behavior[key] or 0.0)
    return 0.0


def _behavior_score(a, b):
    x = _behavior_scalar(a)
    y = _behavior_scalar(b)
    return 1.0 - abs(x - y) / max(x, y, 1.0)


def score_pair(a, b, config):
    sem = max(0.0, cosine(a.get("semantic_mean") or [], b.get("semantic_mean") or []))
    rec = max(0.0, cosine(a.get("semantic_recency") or [], b.get("semantic_recency") or []))
    act = _activity_score(a, b)
    tmp = _temporal_score(a, b)
    beh = _behavior_score(a, b)
    w = _weights(config)
    full = (
        float(w.get("semantic", 0.46)) * sem
        + float(w.get("recency_semantic", 0.14)) * rec
        + float(w.get("activity", 0.14)) * act
        + float(w.get("temporal", 0.16)) * tmp
        + float(w.get("behavior", 0.10)) * beh
    )
    return round(full, 6), {
        "semantic": round(sem, 6),
        "recency_semantic": round(rec, 6),
        "activity": round(act, 6),
        "temporal": round(tmp, 6),
        "behavior": round(beh, 6),
        "semantic_only": round(sem, 6),
        "activity_temporal": round(0.5 * act + 0.5 * tmp, 6),
        "full": round(full, 6),
    }


def top_matches_v2(left, right, config):
    top_k = int(config["pipeline"].get("top_k_matches", 8))
    blocks = defaultdict(list)
    for p in right:
        blocks[block_key(p, config)].append(p)
    fallback = right[:1200]
    rows, counts = [], []
    for lp in left:
        candidates = blocks.get(block_key(lp, config)) or fallback
        counts.append(len(candidates))
        scored = []
        for rp in candidates:
            score, parts = score_pair(lp, rp, config)
            scored.append((score, rp, parts))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, rp, parts) in enumerate(scored[:top_k], 1):
            rows.append({
                "left_dataset": lp["dataset"],
                "left_user_id": lp["user_id"],
                "right_dataset": rp["dataset"],
                "right_user_id": rp["user_id"],
                "rank": rank,
                "score": score,
                "confidence_level": confidence(score, config),
                "score_parts": parts,
                "ablation_semantic_only": parts["semantic_only"],
                "ablation_activity_temporal": parts["activity_temporal"],
                "ablation_full": parts["full"],
                "block_key": block_key(lp, config),
                "candidate_count": len(candidates),
            })
    return rows, counts


def run_matching_v2(config):
    profiles = read_table(os.path.join(config["paths"]["processed"], "user_profiles_v2.parquet"))
    by_ds = defaultdict(list)
    for p in profiles:
        by_ds[p["dataset"]].append(p)
    pairs = [("movielens", "goodreads"), ("mind", "kuairec"), ("movielens", "mind"), ("movielens", "kuairec")]
    results = []
    lines = []
    for left, right in pairs:
        matches, counts = top_matches_v2(by_ds[left], by_ds[right], config)
        out = os.path.join(config["paths"]["processed"], "matches_v2_%s_%s.parquet" % (left, right))
        write_table(out, matches, ["left_dataset", "left_user_id", "right_dataset", "right_user_id", "score", "confidence_level", "score_parts"], config["pipeline"].get("storage_format", "auto"))
        conf = Counter(m["confidence_level"] for m in matches)
        dist = Counter(counts)
        lines.append("- %s_%s rows=%s confidence=%s candidate_distribution=%s" % (left, right, len(matches), dict(conf), dict(dist.most_common(8))))
        lines.append("  samples=`%s`" % matches[:2])
        results.append({"pair": "%s_%s" % (left, right), "path": out, "rows": len(matches), "confidence": dict(conf), "candidate_counts": counts, "sample": matches[:3]})
    write_markdown("reports/v2_matching_upgrade.md", "V2 Matching Upgrade", [("Smoke Test", lines)])
    return results
