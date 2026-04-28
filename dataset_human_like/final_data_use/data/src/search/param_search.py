import argparse
import json
import os
import random
from collections import Counter, defaultdict

import yaml

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io_utils import ensure_dir, load_config, read_json, read_table, table_exists, write_json, write_markdown, write_table
from src.matching import matcher_v2fix


OUT = "data/processed/search"
PAIRS = [("goodreads", "movielens_goodreads"), ("mind", "movielens_mind"), ("kuairec", "movielens_kuairec")]


def _write_stage(stage, objective, command, summary, files, smoke, error="none"):
    log = "logs/stage_search_%s.log" % stage
    ckpt = "reports/checkpoint_search_stage_%s.md" % stage
    ensure_dir(os.path.dirname(log))
    ensure_dir(os.path.dirname(ckpt))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend("- %s" % f for f in files)
    body.append("Smoke test result: %s" % smoke)
    body.append("Error summary: %s" % error)
    with open(log, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = ["# Checkpoint Search Stage %s" % stage, "", "## Objective", "", objective, "", "## Execution Commands", "", "`%s`" % command, "", "## Output Summary", "", summary, "", "## Generated Files", ""]
    md.extend("- `%s`" % f for f in files)
    md.extend(["", "## Smoke Test Result", "", smoke, "", "## Error Summary", "", error, ""])
    with open(ckpt, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def load_search_config(path="configs/pseudo_user_search.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = load_config(cfg["base_config"])
    return cfg, base


def normalize_weights(weights):
    total = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
    return {k: round(max(0.0, float(v)) / total, 6) for k, v in weights.items()}


def sample_params(rng, cfg):
    ranges = cfg["search"]["search_ranges"]
    w = {k: rng.uniform(v[0], v[1]) for k, v in ranges["weights"].items()}
    strict = rng.uniform(*ranges["strict_threshold"])
    medium = rng.uniform(*ranges["medium_threshold"])
    if medium >= strict:
        medium = max(0.40, strict - rng.uniform(0.05, 0.15))
    return {
        "weights": normalize_weights(w),
        "strict_threshold": round(strict, 6),
        "medium_threshold": round(medium, 6),
        "top_k": rng.choice(ranges["top_k"]),
        "fallback_candidate_size": rng.choice(ranges["fallback_candidate_size"]),
    }


def _profiles_by_domain(base):
    rows = read_table(os.path.join(base["paths"]["processed"], "user_profiles_v2fix.parquet"))
    by = defaultdict(list)
    for row in rows:
        by[row["dataset"]].append(row)
    return by


def build_devset(cfg_path="configs/pseudo_user_search.yaml"):
    cfg, base = load_search_config(cfg_path)
    rng = random.Random(int(cfg["search"]["seed"]))
    by = _profiles_by_domain(base)
    anchors = sorted([p["user_id"] for p in by["movielens"]])
    rng.shuffle(anchors)
    anchors = sorted(anchors[: int(cfg["search"]["dev_anchor_count"])])
    out = os.path.join(cfg["search"]["output_dir"], "dev_anchors.json")
    write_json(out, {"seed": cfg["search"]["seed"], "anchor_count": len(anchors), "anchors": anchors})
    write_markdown("reports/search_devset.md", "Search Dev Set", [("Dev Anchors", ["anchor_count=%s" % len(anchors), "sample=%s" % anchors[:10], "selection=fixed random sample from MovieLens V2-fix profile anchors"])])
    return out, anchors


def _candidate_parts(lp, right_profiles, base, fallback_size):
    blocks = defaultdict(list)
    for rp in right_profiles:
        blocks[matcher_v2fix.block_key(rp, base)].append(rp)
    candidates = blocks.get(matcher_v2fix.block_key(lp, base))
    if not candidates:
        candidates = matcher_v2fix._fallback_candidates(lp, right_profiles, {"pipeline": {"fallback_candidates": fallback_size, "semantic_coarse_block": True}})
    return candidates


def precompute_candidates(cfg, base, anchors):
    by = _profiles_by_domain(base)
    anchor_set = set(anchors)
    ml = {p["user_id"]: p for p in by["movielens"] if p["user_id"] in anchor_set}
    max_fb = max(cfg["search"]["search_ranges"]["fallback_candidate_size"])
    candidate_rows = []
    for right_domain, pair_name in PAIRS:
        for uid in anchors:
            lp = ml.get(uid)
            if not lp:
                continue
            for rp in _candidate_parts(lp, by[right_domain], base, max_fb):
                _, parts = matcher_v2fix.score_pair(lp, rp, base)
                candidate_rows.append({
                    "left_user_id": uid,
                    "right_dataset": right_domain,
                    "right_user_id": rp["user_id"],
                    "score_parts": parts,
                    "base_block_key": matcher_v2fix.block_key(lp, base),
                })
    out = os.path.join(cfg["search"]["output_dir"], "dev_candidate_parts.parquet")
    write_table(out, candidate_rows, ["left_user_id", "right_dataset", "right_user_id", "score_parts"])
    return candidate_rows, out


def _score_from_parts(parts, weights):
    return (
        weights["semantic"] * parts["semantic"]
        + weights["recency_semantic"] * parts["recency_semantic"]
        + weights["activity"] * parts["activity"]
        + weights["temporal"] * parts["temporal"]
        + weights["behavior"] * parts["behavior"]
    )


def _confidence(score, params):
    if score >= params["strict_threshold"]:
        return "strict"
    if score >= params["medium_threshold"]:
        return "medium"
    return "loose"


def run_trial(trial_id, params, candidate_rows, cfg):
    top_k = int(params["top_k"])
    min_component = float(params["medium_threshold"])
    min_global = float(params["medium_threshold"])
    by_anchor_domain = defaultdict(list)
    for row in candidate_rows:
        score = _score_from_parts(row["score_parts"], params["weights"])
        r = dict(row)
        r["score"] = round(score, 6)
        r["confidence_level"] = _confidence(score, params)
        by_anchor_domain[(row["left_user_id"], row["right_dataset"])].append(r)
    for key in list(by_anchor_domain.keys()):
        by_anchor_domain[key].sort(key=lambda r: r["score"], reverse=True)
        by_anchor_domain[key] = by_anchor_domain[key][:top_k]
    metadata = []
    anchors = sorted({k[0] for k in by_anchor_domain})
    for uid in anchors:
        selected = []
        for domain, _ in PAIRS:
            cands = [r for r in by_anchor_domain.get((uid, domain), []) if r["score"] >= min_component]
            if cands:
                selected.append((domain, cands[0]))
        selected.sort(key=lambda x: x[1]["score"], reverse=True)
        if len(selected) < 2:
            continue
        chosen = selected[:3] if len(selected) >= 3 else selected[:2]
        gscore = sum(r["score"] for _, r in chosen) / len(chosen)
        if gscore < min_global:
            continue
        members = {"movielens": uid}
        parts_acc = defaultdict(list)
        for domain, r in chosen:
            members[domain] = r["right_user_id"]
            for k, v in r["score_parts"].items():
                parts_acc[k].append(v)
        metadata.append({
            "pseudo_user_id": "trial_%03d_%06d" % (trial_id, len(metadata) + 1),
            "source_members": members,
            "domains_present": sorted(members.keys()),
            "global_consistency_score": round(gscore, 6),
            "confidence_level": _confidence(gscore, params),
            "parts": {k: sum(v) / len(v) for k, v in parts_acc.items()},
        })
    conf = Counter(m["confidence_level"] for m in metadata)
    reuse = Counter()
    vals = defaultdict(list)
    for m in metadata:
        vals["global_consistency"].append(m["global_consistency_score"])
        for k, v in m.get("parts", {}).items():
            vals[k].append(v)
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                reuse[(ds, uid)] += 1
    reused = sum(1 for c in reuse.values() if c > 1)
    reuse_rate = reused / float(len(reuse) or 1)
    count = len(metadata)
    strict = conf.get("strict", 0)
    medium = conf.get("medium", 0)
    loose = conf.get("loose", 0)
    loose_ratio = loose / float(count or 1)
    strict_ratio = strict / float(count or 1)
    def avg(key):
        xs = vals.get(key, [])
        return round(sum(xs) / len(xs), 6) if xs else 0.0
    objective = avg("global_consistency") - 0.5 * loose_ratio - 0.2 * reuse_rate + 0.1 * strict_ratio
    valid = count >= int(cfg["search"]["min_pseudo_user_count"]) and strict >= int(cfg["search"]["min_strict_count"])
    if not valid:
        objective -= 1.0
    return {
        "trial_id": trial_id,
        "params": params,
        "pseudo_user_count": count,
        "strict_count": strict,
        "medium_count": medium,
        "loose_count": loose,
        "global_consistency": avg("global_consistency"),
        "semantic_consistency": avg("semantic"),
        "temporal_consistency": avg("temporal"),
        "behavior_consistency": avg("behavior"),
        "reuse_rate": round(reuse_rate, 6),
        "loose_ratio": round(loose_ratio, 6),
        "strict_ratio": round(strict_ratio, 6),
        "objective": round(objective, 6),
        "valid": valid,
    }


def stage_1(cfg_path):
    cfg, base = load_search_config(cfg_path)
    ensure_dir(cfg["search"]["output_dir"])
    dev_path, anchors = build_devset(cfg_path)
    candidate_rows, cand_path = precompute_candidates(cfg, base, anchors[:100])
    rng = random.Random(int(cfg["search"]["seed"]))
    params = sample_params(rng, cfg)
    result = run_trial(1, params, candidate_rows, cfg)
    trials_path = os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl")
    with open(trials_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
    write_markdown("reports/search_framework.md", "Search Framework", [("Smoke Test", ["trial=%s" % result, "dev_path=%s" % dev_path, "candidate_parts=%s" % cand_path])])
    files = [trials_path, dev_path, cand_path, "reports/search_framework.md"]
    ok = table_exists(trials_path)
    smoke = "ok - one trial completed objective=%s valid=%s" % (result["objective"], result["valid"])
    _write_stage(1, "Implement random-search framework and run one smoke trial.", "python3 src/search/param_search.py --stage 1", "Search framework sampled parameters, precomputed candidates, ran one trial, and wrote JSONL.", files + ["logs/stage_search_1.log", "reports/checkpoint_search_stage_1.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 1 failed")
    print(smoke)


def stage_2(cfg_path):
    cfg, _ = load_search_config(cfg_path)
    out, anchors = build_devset(cfg_path)
    ok = table_exists(out) and table_exists("reports/search_devset.md")
    smoke = "ok - dev_anchor_count=%s sample=%s" % (len(anchors), anchors[:5])
    _write_stage(2, "Define deterministic dev anchor subset for parameter search.", "python3 src/search/param_search.py --stage 2", "Selected fixed MovieLens anchor subset for reproducible search.", [cfg_path, out, "reports/search_devset.md", "logs/stage_search_2.log", "reports/checkpoint_search_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 2 failed")
    print(smoke)


def stage_3(cfg_path):
    cfg, base = load_search_config(cfg_path)
    ensure_dir(cfg["search"]["output_dir"])
    dev = read_json(os.path.join(cfg["search"]["output_dir"], "dev_anchors.json")) if os.path.exists(os.path.join(cfg["search"]["output_dir"], "dev_anchors.json")) else {"anchors": build_devset(cfg_path)[1]}
    candidate_rows, cand_path = precompute_candidates(cfg, base, dev["anchors"])
    rng = random.Random(int(cfg["search"]["seed"]))
    trials_path = os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl")
    results = []
    with open(trials_path, "w", encoding="utf-8") as f:
        for i in range(1, int(cfg["search"]["num_trials"]) + 1):
            params = sample_params(rng, cfg)
            result = run_trial(i, params, candidate_rows, cfg)
            results.append(result)
            f.write(json.dumps(result, sort_keys=True) + "\n")
    best = max(results, key=lambda r: r["objective"])
    summary = {"num_trials": len(results), "best_trial": best, "valid_trials": sum(1 for r in results if r["valid"])}
    summary_path = os.path.join(cfg["search"]["output_dir"], "search_trials_summary.json")
    write_json(summary_path, summary)
    lines = ["first_5=`%s`" % results[:5], "best=`%s`" % best, "candidate_parts=%s" % cand_path]
    write_markdown("reports/search_results_raw.md", "Raw Search Results", [("Smoke Test", lines)])
    ok = table_exists(trials_path) and len(results) >= 50
    smoke = "ok - trials=%s valid=%s best_objective=%s" % (len(results), summary["valid_trials"], best["objective"])
    _write_stage(3, "Run random parameter search.", "python3 src/search/param_search.py --stage 3", "Executed random search and wrote all trial records.", [trials_path, summary_path, "reports/search_results_raw.md", cand_path, "logs/stage_search_3.log", "reports/checkpoint_search_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 3 failed")
    print(smoke)


def stage_4(cfg_path):
    cfg, base = load_search_config(cfg_path)
    trials = []
    with open(os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    top = sorted(trials, key=lambda r: r["objective"], reverse=True)[:5]
    best = top[0]
    tuned = dict(base)
    tuned["pipeline"] = dict(base["pipeline"])
    tuned["pipeline"]["matching_weights"] = best["params"]["weights"]
    tuned["pipeline"]["confidence_thresholds"] = {"strict": best["params"]["strict_threshold"], "medium": best["params"]["medium_threshold"]}
    tuned["pipeline"]["top_k_matches"] = best["params"]["top_k"]
    tuned["pipeline"]["fallback_candidates"] = best["params"]["fallback_candidate_size"]
    tuned["paths"] = dict(base["paths"])
    best_cfg = "configs/pseudo_user_pipeline_v2fix_tuned.yaml"
    with open(best_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False, allow_unicode=True)
    default = read_json("data/processed/eval_summary_v2fix.json")
    lines = ["default_global=%s" % default["v2fix_full_method"]["global_consistency"], "top_5:"]
    lines.extend(["- `%s`" % r for r in top])
    lines.append("best_config=%s" % best_cfg)
    write_markdown("reports/search_best_config.md", "Search Best Config", [("Top Trials", lines)])
    ok = table_exists(best_cfg)
    smoke = "ok - best_trial=%s objective=%s" % (best["trial_id"], best["objective"])
    _write_stage(4, "Select best search trial and write tuned config.", "python3 src/search/param_search.py --stage 4", "Selected top-5 trials and wrote tuned V2-fix config.", [best_cfg, "reports/search_best_config.md", "logs/stage_search_4.log", "reports/checkpoint_search_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 4 failed")
    print(smoke)


def _score_metadata(metadata, profiles, config):
    vals = defaultdict(list)
    for m in metadata:
        anchor = profiles.get(("movielens", m["source_members"].get("movielens")))
        if not anchor:
            continue
        pair_scores = []
        for ds, uid in m["source_members"].items():
            if ds == "movielens":
                continue
            prof = profiles.get((ds, uid))
            if not prof:
                continue
            full, parts = matcher_v2fix.score_pair(anchor, prof, config)
            pair_scores.append(full)
            vals["semantic_consistency"].append(parts["semantic"])
            vals["temporal_consistency"].append(parts["temporal"])
            vals["behavior_consistency"].append(parts["behavior"])
            vals["ablation_full"].append(parts["full"])
        if pair_scores:
            vals["global_consistency"].append(sum(pair_scores) / len(pair_scores))
    return {k: round(sum(v) / len(v), 6) if v else 0.0 for k, v in vals.items()}


def _write_tuned_outputs(config):
    # Run matching with tuned config, then synthesize to temporary normal V2fix names, finally rename.
    from src.matching.matcher_v2fix import run_matching_v2fix
    from src.synthesis.pseudo_users_v2fix import synthesize_v2fix
    run_matching_v2fix(config)
    synthesize_v2fix(config)
    proc = config["paths"]["processed"]
    paths = {}
    for name in ["metadata", "interactions"]:
        src = os.path.join(proc, "pseudo_user_%s_v2fix.parquet" % name)
        dst = os.path.join(proc, "pseudo_user_%s_v2fix_tuned.parquet" % name)
        os.replace(src, dst)
        paths[name] = dst
    return paths


def stage_5(cfg_path):
    tuned = load_config("configs/pseudo_user_pipeline_v2fix_tuned.yaml")
    paths = _write_tuned_outputs(tuned)
    meta = read_table(paths["metadata"])
    interactions_count = sum(1 for _ in open(paths["interactions"], "r", encoding="utf-8")) - 1
    profiles = {(p["dataset"], p["user_id"]): p for p in read_table(os.path.join(tuned["paths"]["processed"], "user_profiles_v2fix.parquet"))}
    metrics = _score_metadata(meta, profiles, tuned)
    conf = Counter(m["confidence_level"] for m in meta)
    reuse = Counter()
    for m in meta:
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                reuse[(ds, uid)] += 1
    reuse_rate = round(sum(1 for c in reuse.values() if c > 1) / float(len(reuse) or 1), 6)
    summary = {"pseudo_user_count": len(meta), "pseudo_interaction_count": interactions_count, "confidence_distribution": dict(conf), "reuse_rate": reuse_rate, "tuned_full_method": metrics, "config": "configs/pseudo_user_pipeline_v2fix_tuned.yaml"}
    out = os.path.join(tuned["paths"]["processed"], "eval_summary_v2fix_tuned.json")
    write_json(out, summary)
    ok = table_exists(paths["metadata"]) and table_exists(paths["interactions"]) and table_exists(out)
    smoke = "ok - pseudo_users=%s interactions=%s confidence=%s global=%s" % (len(meta), interactions_count, dict(conf), metrics.get("global_consistency"))
    _write_stage(5, "Run full tuned V2-fix generation with best parameters.", "python3 src/search/param_search.py --stage 5", "Generated tuned metadata/interactions/eval without overwriting original V2-fix tuned filenames.", [paths["metadata"], paths["interactions"], out, "logs/stage_search_5.log", "reports/checkpoint_search_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 5 failed")
    print(smoke)


def _objective_from_summary(summary):
    count = summary["pseudo_user_count"]
    conf = summary["confidence_distribution"]
    loose = conf.get("loose", 0)
    strict = conf.get("strict", 0)
    loose_ratio = loose / float(count or 1)
    strict_ratio = strict / float(count or 1)
    reuse = summary["reuse_rate"]
    glob = summary["tuned_full_method"]["global_consistency"] if "tuned_full_method" in summary else summary["v2fix_full_method"]["global_consistency"]
    return round(glob - 0.5 * loose_ratio - 0.2 * reuse + 0.1 * strict_ratio, 6)


def stage_6(cfg_path):
    default_raw = read_json("data/processed/eval_summary_v2fix.json")
    default = {"pseudo_user_count": default_raw["pseudo_user_count"], "pseudo_interaction_count": default_raw["pseudo_interaction_count"], "confidence_distribution": default_raw["confidence_distribution"], "reuse_rate": default_raw["match_reuse_rate"], "v2fix_full_method": default_raw["v2fix_full_method"]}
    tuned = read_json("data/processed/eval_summary_v2fix_tuned.json")
    comparison = {"default": default, "tuned": tuned, "default_objective": _objective_from_summary(default), "tuned_objective": _objective_from_summary(tuned)}
    comparison["tuned_better"] = comparison["tuned_objective"] > comparison["default_objective"]
    comparison["recommend_replace_default"] = comparison["tuned_better"] and tuned["pseudo_user_count"] >= 1000
    out = os.path.join(OUT, "search_final_comparison.json")
    write_json(out, comparison)
    lines = [
        "| version | objective | global | users | interactions | confidence | reuse |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        "| default | %s | %s | %s | %s | `%s` | %s |" % (comparison["default_objective"], default["v2fix_full_method"]["global_consistency"], default["pseudo_user_count"], default["pseudo_interaction_count"], default["confidence_distribution"], default["reuse_rate"]),
        "| tuned | %s | %s | %s | %s | `%s` | %s |" % (comparison["tuned_objective"], tuned["tuned_full_method"]["global_consistency"], tuned["pseudo_user_count"], tuned["pseudo_interaction_count"], tuned["confidence_distribution"], tuned["reuse_rate"]),
        "",
        "tuned_better=%s" % comparison["tuned_better"],
        "recommend_replace_default=%s" % comparison["recommend_replace_default"],
    ]
    write_markdown("reports/search_comparison.md", "Search Final Comparison", [("Comparison", lines)])
    ok = table_exists(out) and table_exists("reports/search_comparison.md")
    smoke = "ok - tuned_better=%s recommend_replace_default=%s" % (comparison["tuned_better"], comparison["recommend_replace_default"])
    _write_stage(6, "Compare default V2-fix and tuned V2-fix.", "python3 src/search/param_search.py --stage 6", "Wrote final tuned-vs-default comparison.", [out, "reports/search_comparison.md", "logs/stage_search_6.log", "reports/checkpoint_search_stage_6.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search stage 6 failed")
    print(smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", default="configs/pseudo_user_search.yaml")
    args = parser.parse_args()
    stages = {1: stage_1, 2: stage_2, 3: stage_3, 4: stage_4, 5: stage_5, 6: stage_6}
    stages[args.stage](args.config)


if __name__ == "__main__":
    main()
