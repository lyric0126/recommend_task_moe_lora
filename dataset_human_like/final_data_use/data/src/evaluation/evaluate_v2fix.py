import json
import os
import random
from collections import Counter, defaultdict

from src.io_utils import read_json, read_table, write_json, write_markdown
from src.matching.matcher_v2fix import score_pair


def _profiles(config):
    rows = read_table(os.path.join(config["paths"]["processed"], "user_profiles_v2fix.parquet"))
    return {(r["dataset"], r["user_id"]): r for r in rows}


def _score(metadata, profiles, config):
    vals = defaultdict(list)
    for meta in metadata:
        anchor = profiles.get(("movielens", meta["source_members"].get("movielens")))
        if not anchor:
            continue
        pair_scores = []
        for ds, uid in meta["source_members"].items():
            if ds == "movielens":
                continue
            prof = profiles.get((ds, uid))
            if not prof:
                continue
            full, parts = score_pair(anchor, prof, config)
            pair_scores.append(full)
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
    rng = random.Random(41)
    by_ds = defaultdict(list)
    for ds, uid in profiles:
        by_ds[ds].append(uid)
    out = []
    for meta in metadata:
        members = {"movielens": meta["source_members"].get("movielens")}
        for ds in ["goodreads", "mind", "kuairec"]:
            if ds in meta["source_members"] and by_ds[ds]:
                members[ds] = rng.choice(by_ds[ds])
        out.append({"source_members": members})
    return out


def evaluate_v2fix(config):
    metadata = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_metadata_v2fix.parquet"))
    interactions = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_interactions_v2fix.parquet"))
    profiles = _profiles(config)
    full = _score(metadata, profiles, config)
    random_mix = _score(_random_mix(metadata, profiles), profiles, config)
    v1 = read_json(os.path.join(config["paths"]["processed"], "eval_summary.json"))
    v2 = read_json(os.path.join(config["paths"]["processed"], "eval_summary_v2.json"))
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    conf = Counter(m["confidence_level"] for m in metadata)
    by_conf = {}
    for level in ["strict", "medium", "loose"]:
        subset = [m for m in metadata if m["confidence_level"] == level]
        by_conf[level] = _score(subset, profiles, config) if subset else {}
    reuse = Counter()
    for m in metadata:
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                reuse[(ds, uid)] += 1
    reused = sum(1 for count in reuse.values() if count > 1)
    summary = {
        "v1_global": v1["full_method"]["global_consistency"],
        "v2_global": v2["v2_full_method"]["global_consistency"],
        "v2fix_full_method": full,
        "v2fix_random_mix": random_mix,
        "v2fix_better_than_v2": full.get("global_consistency", 0) > v2["v2_full_method"]["global_consistency"],
        "v2fix_close_to_or_above_v1": full.get("global_consistency", 0) >= v1["full_method"]["global_consistency"] * 0.9,
        "full_better_than_semantic_only": full.get("ablation_full", 0) > full.get("ablation_semantic_only", 0),
        "full_vs_activity_temporal_gap": round(full.get("ablation_full", 0) - full.get("ablation_activity_temporal", 0), 6),
        "domain_coverage_distribution": {str(k): v for k, v in coverage.items()},
        "confidence_distribution": dict(conf),
        "confidence_level_metrics": by_conf,
        "match_reuse_rate": round(reused / float(len(reuse) or 1), 6),
        "pseudo_user_count": len(metadata),
        "pseudo_interaction_count": len(interactions),
        "ablation": {
            "semantic_only": full.get("ablation_semantic_only", 0),
            "activity_temporal": full.get("ablation_activity_temporal", 0),
            "full": full.get("ablation_full", 0),
        },
        "v1": v1,
        "v2": v2,
    }
    out = os.path.join(config["paths"]["processed"], "eval_summary_v2fix.json")
    write_json(out, summary)
    lines = [
        "| method | global | semantic | temporal | behavior | pseudo_users | confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        "| Random Mix V1 | %s | %s | %s | %s | - | - |" % (v1["random_mix"]["global_consistency"], v1["random_mix"]["semantic_consistency"], v1["random_mix"]["temporal_consistency"], v1["random_mix"]["behavior_consistency"]),
        "| V1 | %s | %s | %s | %s | %s | - |" % (v1["full_method"]["global_consistency"], v1["full_method"]["semantic_consistency"], v1["full_method"]["temporal_consistency"], v1["full_method"]["behavior_consistency"], v1["pseudo_user_count"]),
        "| V2 | %s | %s | %s | %s | %s | %s |" % (v2["v2_full_method"]["global_consistency"], v2["v2_full_method"]["semantic_consistency"], v2["v2_full_method"]["temporal_consistency"], v2["v2_full_method"]["behavior_consistency"], v2["pseudo_user_count"], v2["confidence_distribution"]),
        "| V2-fix Random Mix | %s | %s | %s | %s | - | - |" % (random_mix.get("global_consistency"), random_mix.get("semantic_consistency"), random_mix.get("temporal_consistency"), random_mix.get("behavior_consistency")),
        "| V2-fix | %s | %s | %s | %s | %s | %s |" % (full.get("global_consistency"), full.get("semantic_consistency"), full.get("temporal_consistency"), full.get("behavior_consistency"), len(metadata), dict(conf)),
        "",
        "V2-fix better than V2: `%s`" % summary["v2fix_better_than_v2"],
        "V2-fix close to or above V1: `%s`" % summary["v2fix_close_to_or_above_v1"],
        "Full better than semantic-only: `%s`" % summary["full_better_than_semantic_only"],
        "Full minus activity+temporal: `%s`" % summary["full_vs_activity_temporal_gap"],
        "Coverage: `%s`" % summary["domain_coverage_distribution"],
        "Ablation: `%s`" % json.dumps(summary["ablation"], sort_keys=True),
        "Match reuse rate: `%s`" % summary["match_reuse_rate"],
    ]
    write_markdown("reports/pseudo_user_eval_v2fix.md", "Pseudo User Evaluation V2fix", [("Comparison", lines)])
    return {"path": out, "summary": summary}
