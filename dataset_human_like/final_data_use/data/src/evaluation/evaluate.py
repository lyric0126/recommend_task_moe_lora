import json
import os
import random
from collections import Counter, defaultdict

from src.features.profiles import cosine
from src.io_utils import read_table, write_json, write_markdown
from src.matching.matcher import pair_score


def _profiles_by_key(config):
    profiles = read_table(os.path.join(config["paths"]["processed"], "user_profiles.parquet"))
    return {(p["dataset"], p["user_id"]): p for p in profiles}


def _score_metadata(metadata, profiles):
    semantic_scores = []
    temporal_scores = []
    behavior_scores = []
    global_scores = []
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
            score, parts = pair_score(anchor, prof)
            semantic_scores.append(parts["semantic"])
            temporal_scores.append(parts["temporal"])
            behavior_scores.append(parts["behavior"])
            pair_scores.append(score)
        if pair_scores:
            global_scores.append(sum(pair_scores) / len(pair_scores))
    def avg(vals):
        return round(sum(vals) / len(vals), 6) if vals else 0.0
    return {
        "semantic_consistency": avg(semantic_scores),
        "temporal_consistency": avg(temporal_scores),
        "behavior_consistency": avg(behavior_scores),
        "global_consistency": avg(global_scores),
    }


def _random_mix(metadata, profiles):
    by_ds = defaultdict(list)
    for (ds, uid), prof in profiles.items():
        by_ds[ds].append(uid)
    rng = random.Random(13)
    baseline = []
    for meta in metadata:
        source = {"movielens": meta["source_members"].get("movielens")}
        for ds in ["goodreads", "mind", "kuairec"]:
            if by_ds[ds]:
                source[ds] = rng.choice(by_ds[ds])
        baseline.append({"source_members": source})
    return baseline


def evaluate(config):
    metadata = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_metadata.parquet"))
    pseudo_interactions = read_table(os.path.join(config["paths"]["processed"], "pseudo_user_interactions.parquet"))
    profiles = _profiles_by_key(config)
    full = _score_metadata(metadata, profiles)
    random_mix = _score_metadata(_random_mix(metadata, profiles), profiles)
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    match_reuse = Counter()
    for m in metadata:
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                match_reuse[(ds, uid)] += 1
    reused = sum(1 for _, count in match_reuse.items() if count > 1)
    summary = {
        "full_method": full,
        "random_mix": random_mix,
        "domain_coverage_distribution": {str(k): v for k, v in coverage.items()},
        "match_reuse_rate": round(reused / float(len(match_reuse) or 1), 6),
        "pseudo_user_count": len(metadata),
        "pseudo_interaction_count": len(pseudo_interactions),
    }
    out = os.path.join(config["paths"]["processed"], "eval_summary.json")
    write_json(out, summary)
    lines = [
        "Random Mix: `%s`" % json.dumps(random_mix, sort_keys=True),
        "Full Method: `%s`" % json.dumps(full, sort_keys=True),
        "Domain coverage: `%s`" % summary["domain_coverage_distribution"],
        "Match reuse rate: `%s`" % summary["match_reuse_rate"],
    ]
    write_markdown("reports/pseudo_user_eval.md", "Pseudo User Evaluation", [("Comparison", lines)])
    return {"path": out, "summary": summary}
