import json
import os
import random
from collections import Counter, defaultdict

from src.io_utils import read_json, read_table, write_json, write_markdown
from src.matching.matcher_v2 import score_pair


def _profiles(config):
    rows = read_table(os.path.join(config["paths"]["processed"], "user_profiles_v2.parquet"))
    return {(r["dataset"], r["user_id"]): r for r in rows}


def _score_meta(metadata, profiles, config):
    vals = defaultdict(list)
    for meta in metadata:
        members = meta["source_members"]
        anchor = profiles.get(("movielens", members.get("movielens")))
        if not anchor:
            continue
        pair_scores = []
        for ds, uid in members.items():
            if ds == "movielens":
                continue
            prof = profiles.get((ds, uid))
            if not prof:
                continue
            score, parts = score_pair(anchor, prof, config)
            pair_scores.append(score)
            vals["semantic_consistency"].append(parts["semantic"])
            vals["temporal_consistency"].append(parts["temporal"])
            vals["behavior_consistency"].append(parts["behavior"])
            vals["ablation_semantic_only"].append(parts["semantic_only"])
            vals["ablation_activity_temporal"].append(parts["activity_temporal"])
            vals["ablation_full"].append(parts["full"])
        if pair_scores:
            vals["global_consistency"].append(sum(pair_scores) / len(pair_scores))
    return {k: round(sum(v) / len(v), 6) if v else 0.0 for k, v in vals.items()}


def _random_mix(metadata, profiles):
    rng = random.Random(29)
    by_ds = defaultdict(list)
    for ds_uid in profiles:
        by_ds[ds_uid[0]].append(ds_uid[1])
    out = []
    for meta in metadata:
        source = {"movielens": meta["source_members"].get("movielens")}
        domains = [d for d in ["goodreads", "mind", "kuairec"] if d in meta["source_members"]]
        for ds in domains:
            if by_ds[ds]:
                source[ds] = rng.choice(by_ds[ds])
        out.append({"source_members": source})
    return out


def evaluate_v2(config):
    metadata = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_metadata_v2.parquet"))
    interactions = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_interactions_v2.parquet"))
    profiles = _profiles(config)
    full = _score_meta(metadata, profiles, config)
    baseline = _score_meta(_random_mix(metadata, profiles), profiles, config)
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    confidence = Counter(m["confidence_level"] for m in metadata)
    by_conf = {}
    for level in ["strict", "medium", "loose"]:
        subset = [m for m in metadata if m["confidence_level"] == level]
        by_conf[level] = _score_meta(subset, profiles, config) if subset else {}
    reuse = Counter()
    for m in metadata:
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                reuse[(ds, uid)] += 1
    reused = sum(1 for count in reuse.values() if count > 1)
    v1 = read_json(os.path.join(config["paths"]["processed"], "eval_summary.json"))
    summary = {
        "v1": v1,
        "v2_full_method": full,
        "v2_random_mix": baseline,
        "domain_coverage_distribution": {str(k): v for k, v in coverage.items()},
        "confidence_distribution": dict(confidence),
        "confidence_level_metrics": by_conf,
        "match_reuse_rate": round(reused / float(len(reuse) or 1), 6),
        "pseudo_user_count": len(metadata),
        "pseudo_interaction_count": len(interactions),
        "v2_better_than_v1_global": full.get("global_consistency", 0) > v1.get("full_method", {}).get("global_consistency", 0),
        "ablation": {
            "semantic_only": full.get("ablation_semantic_only", 0),
            "activity_temporal": full.get("ablation_activity_temporal", 0),
            "full": full.get("ablation_full", 0),
        },
    }
    out = os.path.join(config["paths"]["processed"], "eval_summary_v2.json")
    write_json(out, summary)
    lines = [
        "V1 full global=%s" % v1.get("full_method", {}).get("global_consistency"),
        "V2 full global=%s" % full.get("global_consistency"),
        "V2 better than V1 global=%s" % summary["v2_better_than_v1_global"],
        "Random Mix V2=`%s`" % json.dumps(baseline, sort_keys=True),
        "Full Method V2=`%s`" % json.dumps(full, sort_keys=True),
        "Ablation=`%s`" % json.dumps(summary["ablation"], sort_keys=True),
        "Coverage=`%s`" % summary["domain_coverage_distribution"],
        "Confidence=`%s`" % summary["confidence_distribution"],
        "Match reuse rate=%s" % summary["match_reuse_rate"],
    ]
    write_markdown("reports/pseudo_user_eval_v2.md", "Pseudo User Evaluation V2", [("V1 vs V2", lines)])
    return {"path": out, "summary": summary}
