import os
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_markdown, write_table


def thresholds(config):
    return config["pipeline"].get("confidence_thresholds", {"strict": 0.62, "medium": 0.5})


def confidence(score, config):
    th = thresholds(config)
    if score >= float(th.get("strict", 0.62)):
        return "strict"
    if score >= float(th.get("medium", 0.5)):
        return "medium"
    return "loose"


def _activity_bucket(p):
    n = int(p.get("activity_total_interactions") or 0)
    if n < 35:
        return "low"
    if n < 120:
        return "mid"
    return "high"


def _temporal_bucket(p):
    peak = int(p.get("temporal_peak_hour") or 0)
    if peak < 7:
        return "night"
    if peak < 15:
        return "day"
    return "evening"


def _semantic_bucket(p):
    vec = p.get("semantic_long_term") or []
    if not vec:
        return "sem_none"
    top = sorted(range(len(vec)), key=lambda i: abs(vec[i]), reverse=True)[:2]
    return "sem_" + "_".join(str(i % 16) for i in top)


def block_key(p, config):
    parts = [_activity_bucket(p), _temporal_bucket(p)]
    if config["pipeline"].get("semantic_coarse_block", True):
        parts.append(_semantic_bucket(p))
    return "|".join(parts)


def _activity_score(a, b):
    fields = ["activity_total_interactions", "activity_active_days", "activity_session_count", "activity_avg_session_len"]
    vals = []
    for f in fields:
        x, y = float(a.get(f) or 0), float(b.get(f) or 0)
        vals.append(1.0 - abs(x - y) / max(x, y, 1.0))
    return sum(vals) / len(vals)


def _temporal_score(a, b):
    hour = max(0.0, cosine(a.get("temporal_hour_hist") or [], b.get("temporal_hour_hist") or []))
    dow = max(0.0, cosine(a.get("temporal_dow_hist") or [], b.get("temporal_dow_hist") or []))
    peak_gap = abs(int(a.get("temporal_peak_hour") or 0) - int(b.get("temporal_peak_hour") or 0))
    peak = 1.0 - min(peak_gap, 12) / 12.0
    return 0.65 * hour + 0.20 * dow + 0.15 * peak


def _behavior_scalar(p):
    b = p.get("behavior_summary") or {}
    for k in ["rating_strength", "interaction_intensity", "click_rate", "watch_completion_tendency", "avg_event_value"]:
        if k in b:
            return float(b[k] or 0.0)
    return 0.0


def _behavior_score(a, b):
    x, y = _behavior_scalar(a), _behavior_scalar(b)
    return 1.0 - abs(x - y) / max(x, y, 1.0)


def score_pair(a, b, config):
    sem = max(0.0, cosine(a.get("semantic_long_term") or [], b.get("semantic_long_term") or []))
    rec = max(0.0, cosine(a.get("semantic_recent") or [], b.get("semantic_recent") or []))
    act = _activity_score(a, b)
    tmp = _temporal_score(a, b)
    beh = _behavior_score(a, b)
    w = config["pipeline"].get("matching_weights", {})
    full = (
        float(w.get("semantic", 0.12)) * sem
        + float(w.get("recency_semantic", 0.06)) * rec
        + float(w.get("activity", 0.28)) * act
        + float(w.get("temporal", 0.34)) * tmp
        + float(w.get("behavior", 0.20)) * beh
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


def _fallback_candidates(lp, right, config):
    limit = int(config["pipeline"].get("fallback_candidates", 180))
    ab, tb = _activity_bucket(lp), _temporal_bucket(lp)
    scoped = [p for p in right if _activity_bucket(p) == ab and _temporal_bucket(p) == tb]
    return (scoped or right)[:limit]


def top_matches_v2fix(left, right, config):
    top_k = int(config["pipeline"].get("top_k_matches", 3))
    blocks = defaultdict(list)
    for p in right:
        blocks[block_key(p, config)].append(p)
    rows, counts = [], []
    for lp in left:
        candidates = blocks.get(block_key(lp, config)) or _fallback_candidates(lp, right, config)
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


def run_matching_v2fix(config):
    profiles = read_table(os.path.join(config["paths"]["processed"], "user_profiles_v2fix.parquet"))
    by_ds = defaultdict(list)
    for p in profiles:
        by_ds[p["dataset"]].append(p)
    pairs = [("movielens", "goodreads"), ("mind", "kuairec"), ("movielens", "mind"), ("movielens", "kuairec")]
    results, lines = [], []
    for left, right in pairs:
        matches, counts = top_matches_v2fix(by_ds[left], by_ds[right], config)
        out = os.path.join(config["paths"]["processed"], "matches_v2fix_%s_%s.parquet" % (left, right))
        write_table(out, matches, ["left_dataset", "left_user_id", "right_dataset", "right_user_id", "score", "confidence_level", "score_parts"], config["pipeline"].get("storage_format", "auto"))
        conf = Counter(m["confidence_level"] for m in matches)
        score_bins = Counter(int(float(m["score"]) * 10) / 10.0 for m in matches)
        dist = Counter(counts)
        v2_path = os.path.join(config["paths"]["processed"], "matches_v2_%s_%s.parquet" % (left, right))
        v2_rows = read_table(v2_path) if os.path.exists(v2_path) else []
        lines.append("- %s_%s rows=%s v2_rows=%s confidence=%s candidate_distribution=%s score_bins=%s" % (left, right, len(matches), len(v2_rows), dict(conf), dict(dist.most_common(8)), dict(score_bins)))
        lines.append("  samples=`%s`" % matches[:2])
        results.append({"pair": "%s_%s" % (left, right), "path": out, "rows": len(matches), "confidence": dict(conf), "candidate_counts": counts, "sample": matches[:3]})
    write_markdown("reports/v2fix_matching.md", "V2fix Matching", [("Smoke Test", lines)])
    return results
