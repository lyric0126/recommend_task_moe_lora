import argparse
import json
import os
import random
from collections import Counter

import yaml

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io_utils import ensure_dir, load_config, read_json, read_table, table_exists, write_json, write_markdown
from src.search import param_search as s1


OUT = "data/processed/search_round2"


def _write_stage(stage, objective, command, summary, files, smoke, error="none"):
    log = "logs/stage_search2_%s.log" % stage
    ckpt = "reports/checkpoint_search2_stage_%s.md" % stage
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
    md = ["# Checkpoint Search2 Stage %s" % stage, "", "## Objective", "", objective, "", "## Execution Commands", "", "`%s`" % command, "", "## Output Summary", "", summary, "", "## Generated Files", ""]
    md.extend("- `%s`" % f for f in files)
    md.extend(["", "## Smoke Test Result", "", smoke, "", "## Error Summary", "", error, ""])
    with open(ckpt, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def load_round2(path="configs/pseudo_user_search_round2.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = load_config(cfg["base_config"])
    return cfg, base


def sample_params(rng, cfg):
    ranges = cfg["search"]["search_ranges"]
    w = {k: rng.uniform(v[0], v[1]) for k, v in ranges["weights"].items()}
    strict = rng.uniform(*ranges["strict_threshold"])
    medium = rng.uniform(*ranges["medium_threshold"])
    if medium >= strict:
        medium = max(ranges["medium_threshold"][0], strict - rng.uniform(0.06, 0.12))
    return {
        "weights": s1.normalize_weights(w),
        "strict_threshold": round(strict, 6),
        "medium_threshold": round(medium, 6),
        "top_k": rng.choice(ranges["top_k"]),
        "fallback_candidate_size": rng.choice(ranges["fallback_candidate_size"]),
    }


def build_devset(cfg_path):
    cfg, base = load_round2(cfg_path)
    rng = random.Random(int(cfg["search"]["seed"]))
    by = s1._profiles_by_domain(base)
    anchors = sorted([p["user_id"] for p in by["movielens"]])
    rng.shuffle(anchors)
    anchors = sorted(anchors[: int(cfg["search"]["dev_anchor_count"])])
    ensure_dir(cfg["search"]["output_dir"])
    out = os.path.join(cfg["search"]["output_dir"], "dev_anchors.json")
    write_json(out, {"seed": cfg["search"]["seed"], "anchor_count": len(anchors), "anchors": anchors})
    return out, anchors


def run_trial_round2(trial_id, params, candidate_rows, cfg):
    r = s1.run_trial(trial_id, params, candidate_rows, {
        "search": {
            "min_pseudo_user_count": 0,
            "min_strict_count": 0,
        }
    })
    count = r["pseudo_user_count"]
    strict = r["strict_count"]
    loose = r["loose_count"]
    strict_ratio = strict / float(count or 1)
    estimated_interactions = int(round(count * float(cfg["search"]["avg_interactions_per_pseudo_user"])))
    growth = max(0.0, estimated_interactions / float(cfg["search"]["default_interaction_count"]) - 1.0)
    reasons = []
    if count < int(cfg["search"]["min_pseudo_user_count"]):
        reasons.append("pseudo_user_count_below_min")
    if strict < int(cfg["search"]["min_strict_count"]):
        reasons.append("strict_count_below_min")
    if loose > int(cfg["search"]["max_loose_count"]):
        reasons.append("loose_count_exceeds_limit")
    if estimated_interactions > float(cfg["search"]["default_interaction_count"]) * float(cfg["search"]["max_interaction_growth"]):
        reasons.append("interaction_growth_exceeds_limit")
    objective = (
        r["global_consistency"]
        + 0.15 * strict_ratio
        - 0.15 * r["reuse_rate"]
        - 0.20 * growth
    )
    valid = not reasons
    if not valid:
        objective -= 1.0 + 0.05 * len(reasons)
    r["interaction_count"] = estimated_interactions
    r["interaction_growth_penalty"] = round(growth, 6)
    r["objective"] = round(objective, 6)
    r["valid"] = valid
    r["invalid_reason"] = reasons
    return r


def _prepare_candidates(cfg, base, anchors):
    candidate_rows, cand_path = s1.precompute_candidates(cfg, base, anchors)
    return candidate_rows, cand_path


def stage_1(cfg_path):
    cfg, base = load_round2(cfg_path)
    ensure_dir(cfg["search"]["output_dir"])
    _, anchors = build_devset(cfg_path)
    candidate_rows, cand_path = _prepare_candidates(cfg, base, anchors[:120])
    rng = random.Random(int(cfg["search"]["seed"]))
    result = run_trial_round2(1, sample_params(rng, cfg), candidate_rows, cfg)
    path = os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
    lines = ["trial=`%s`" % result, "valid_logic=%s invalid_reason=%s" % (result["valid"], result["invalid_reason"]), "candidate_parts=%s" % cand_path]
    write_markdown("reports/search_round2_framework.md", "Search Round2 Framework", [("Smoke Test", lines)])
    ok = table_exists(path)
    smoke = "ok - one trial objective=%s valid=%s reasons=%s" % (result["objective"], result["valid"], result["invalid_reason"])
    _write_stage(1, "Configure conservative second-round search framework.", "python3 src/search/param_search_round2.py --stage 1", "Ran one trial and demonstrated valid/invalid logic.", [cfg_path, path, "reports/search_round2_framework.md", "logs/stage_search2_1.log", "reports/checkpoint_search2_stage_1.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search2 stage 1 failed")
    print(smoke)


def stage_2(cfg_path):
    cfg, base = load_round2(cfg_path)
    ensure_dir(cfg["search"]["output_dir"])
    _, anchors = build_devset(cfg_path)
    candidate_rows, cand_path = _prepare_candidates(cfg, base, anchors)
    rng = random.Random(int(cfg["search"]["seed"]))
    path = os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl")
    results = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, int(cfg["search"]["num_trials"]) + 1):
            res = run_trial_round2(i, sample_params(rng, cfg), candidate_rows, cfg)
            results.append(res)
            f.write(json.dumps(res, sort_keys=True) + "\n")
    valid = [r for r in results if r["valid"]]
    best = max(valid or results, key=lambda r: r["objective"])
    reason_counts = Counter(reason for r in results for reason in r.get("invalid_reason", []))
    summary = {"num_trials": len(results), "valid_trials": len(valid), "best_valid_trial": best if best.get("valid") else None, "best_any_trial": max(results, key=lambda r: r["objective"]), "invalid_reason_distribution": dict(reason_counts)}
    summary_path = os.path.join(cfg["search"]["output_dir"], "search_trials_summary.json")
    write_json(summary_path, summary)
    write_markdown("reports/search_round2_results_raw.md", "Search Round2 Raw Results", [("Smoke Test", ["best=`%s`" % best, "invalid_reasons=%s" % dict(reason_counts), "first_5=`%s`" % results[:5]])])
    ok = table_exists(path) and len(results) >= 40
    smoke = "ok - trials=%s valid=%s best_objective=%s invalid_reasons=%s" % (len(results), len(valid), best["objective"], dict(reason_counts))
    _write_stage(2, "Execute conservative second-round random search.", "python3 src/search/param_search_round2.py --stage 2", "Ran round2 search and wrote all trial records.", [path, summary_path, "reports/search_round2_results_raw.md", cand_path, "logs/stage_search2_2.log", "reports/checkpoint_search2_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search2 stage 2 failed")
    print(smoke)


def stage_3(cfg_path):
    cfg, base = load_round2(cfg_path)
    path = os.path.join(cfg["search"]["output_dir"], "search_trials.jsonl")
    trials = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    valid = [r for r in trials if r["valid"]]
    if not valid:
        raise RuntimeError("No valid round2 trials")
    top = sorted(valid, key=lambda r: r["objective"], reverse=True)[:5]
    best = top[0]
    tuned = dict(base)
    tuned["pipeline"] = dict(base["pipeline"])
    tuned["pipeline"]["matching_weights"] = best["params"]["weights"]
    tuned["pipeline"]["confidence_thresholds"] = {"strict": best["params"]["strict_threshold"], "medium": best["params"]["medium_threshold"]}
    tuned["pipeline"]["top_k_matches"] = best["params"]["top_k"]
    tuned["pipeline"]["fallback_candidates"] = best["params"]["fallback_candidate_size"]
    out_cfg = "configs/pseudo_user_pipeline_v2fix_round2_best.yaml"
    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False, allow_unicode=True)
    first = read_json("data/processed/search/search_final_comparison.json")
    lines = ["default_objective=%s" % first["default_objective"], "first_round_tuned_objective=%s" % first["tuned_objective"], "top_5_valid:"]
    lines.extend(["- `%s`" % r for r in top])
    write_markdown("reports/search_round2_best_config.md", "Search Round2 Best Config", [("Top Valid Trials", lines)])
    ok = table_exists(out_cfg)
    smoke = "ok - best_trial=%s objective=%s" % (best["trial_id"], best["objective"])
    _write_stage(3, "Select best valid second-round parameter set.", "python3 src/search/param_search_round2.py --stage 3", "Wrote top-5 valid trials and round2 best config.", [out_cfg, "reports/search_round2_best_config.md", "logs/stage_search2_3.log", "reports/checkpoint_search2_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search2 stage 3 failed")
    print(smoke)


def _restore_default_outputs():
    # Keep original default V2-fix files intact after using the normal V2-fix writer.
    src_meta = "data/final_versions/v2fix_all/pseudo_user_metadata.parquet"
    src_inter = "data/final_versions/v2fix_all/pseudo_user_interactions.parquet"
    if os.path.exists(src_meta):
        import shutil
        shutil.copyfile(src_meta, "data/processed/pseudo_user_metadata_v2fix.parquet")
    if os.path.exists(src_inter):
        import shutil
        shutil.copyfile(src_inter, "data/processed/pseudo_user_interactions_v2fix.parquet")


def stage_4(cfg_path):
    cfg = load_config("configs/pseudo_user_pipeline_v2fix_round2_best.yaml")
    from src.search.param_search import _write_tuned_outputs, _score_metadata
    paths = _write_tuned_outputs(cfg)
    round2_meta = "data/processed/pseudo_user_metadata_v2fix_round2.parquet"
    round2_inter = "data/processed/pseudo_user_interactions_v2fix_round2.parquet"
    os.replace(paths["metadata"], round2_meta)
    os.replace(paths["interactions"], round2_inter)
    _restore_default_outputs()
    meta = read_table(round2_meta)
    interactions_count = sum(1 for _ in open(round2_inter, "r", encoding="utf-8")) - 1
    profiles = {(p["dataset"], p["user_id"]): p for p in read_table("data/processed/user_profiles_v2fix.parquet")}
    metrics = _score_metadata(meta, profiles, cfg)
    conf = Counter(m["confidence_level"] for m in meta)
    reuse = Counter()
    for m in meta:
        for ds, uid in m["source_members"].items():
            if ds != "movielens":
                reuse[(ds, uid)] += 1
    reuse_rate = round(sum(1 for c in reuse.values() if c > 1) / float(len(reuse) or 1), 6)
    summary = {"pseudo_user_count": len(meta), "pseudo_interaction_count": interactions_count, "confidence_distribution": dict(conf), "reuse_rate": reuse_rate, "round2_full_method": metrics, "config": "configs/pseudo_user_pipeline_v2fix_round2_best.yaml"}
    out = "data/processed/eval_summary_v2fix_round2.json"
    write_json(out, summary)
    ok = table_exists(round2_meta) and table_exists(round2_inter) and table_exists(out)
    smoke = "ok - pseudo_users=%s interactions=%s confidence=%s global=%s" % (len(meta), interactions_count, dict(conf), metrics.get("global_consistency"))
    _write_stage(4, "Run full V2-fix with second-round best parameters.", "python3 src/search/param_search_round2.py --stage 4", "Generated round2 metadata/interactions/eval without replacing default or first tuned files.", [round2_meta, round2_inter, out, "logs/stage_search2_4.log", "reports/checkpoint_search2_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search2 stage 4 failed")
    print(smoke)


def _objective(summary, method_key):
    count = summary["pseudo_user_count"]
    conf = summary["confidence_distribution"]
    strict = conf.get("strict", 0)
    loose = conf.get("loose", 0)
    strict_ratio = strict / float(count or 1)
    growth = max(0.0, summary["pseudo_interaction_count"] / 298777.0 - 1.0)
    global_score = summary[method_key]["global_consistency"]
    obj = global_score + 0.15 * strict_ratio - 0.15 * summary["reuse_rate"] - 0.20 * growth
    if loose > 5:
        obj -= 1.0
    return round(obj, 6)


def stage_5(cfg_path):
    default_raw = read_json("data/processed/eval_summary_v2fix.json")
    default = {"pseudo_user_count": default_raw["pseudo_user_count"], "pseudo_interaction_count": default_raw["pseudo_interaction_count"], "confidence_distribution": default_raw["confidence_distribution"], "reuse_rate": default_raw["match_reuse_rate"], "v2fix_full_method": default_raw["v2fix_full_method"]}
    first = read_json("data/processed/eval_summary_v2fix_tuned.json")
    second = read_json("data/processed/eval_summary_v2fix_round2.json")
    comp = {
        "default": default,
        "first_round_tuned": first,
        "second_round": second,
        "default_objective_round2": _objective(default, "v2fix_full_method"),
        "first_round_objective_round2": _objective(first, "tuned_full_method"),
        "second_round_objective": _objective(second, "round2_full_method"),
    }
    comp["round2_more_stable_than_first"] = second["confidence_distribution"].get("loose", 0) <= first["confidence_distribution"].get("loose", 0) and second["pseudo_interaction_count"] <= first["pseudo_interaction_count"]
    comp["round2_beats_default_without_loose"] = comp["second_round_objective"] > comp["default_objective_round2"] and second["confidence_distribution"].get("loose", 0) == 0
    comp["recommendation"] = "upgrade_to_round2" if comp["round2_beats_default_without_loose"] else "keep_default"
    out = os.path.join(OUT, "search_round2_final_comparison.json")
    write_json(out, comp)
    lines = [
        "| version | objective | global | users | interactions | confidence | reuse |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        "| default | %s | %s | %s | %s | `%s` | %s |" % (comp["default_objective_round2"], default["v2fix_full_method"]["global_consistency"], default["pseudo_user_count"], default["pseudo_interaction_count"], default["confidence_distribution"], default["reuse_rate"]),
        "| first tuned | %s | %s | %s | %s | `%s` | %s |" % (comp["first_round_objective_round2"], first["tuned_full_method"]["global_consistency"], first["pseudo_user_count"], first["pseudo_interaction_count"], first["confidence_distribution"], first["reuse_rate"]),
        "| round2 | %s | %s | %s | %s | `%s` | %s |" % (comp["second_round_objective"], second["round2_full_method"]["global_consistency"], second["pseudo_user_count"], second["pseudo_interaction_count"], second["confidence_distribution"], second["reuse_rate"]),
        "",
        "round2_more_stable_than_first=%s" % comp["round2_more_stable_than_first"],
        "round2_beats_default_without_loose=%s" % comp["round2_beats_default_without_loose"],
        "recommendation=%s" % comp["recommendation"],
    ]
    write_markdown("reports/search_round2_comparison.md", "Search Round2 Final Comparison", [("Comparison", lines)])
    ok = table_exists(out) and table_exists("reports/search_round2_comparison.md")
    smoke = "ok - recommendation=%s round2_beats_default_without_loose=%s" % (comp["recommendation"], comp["round2_beats_default_without_loose"])
    _write_stage(5, "Compare default, first tuned, and second-round best.", "python3 src/search/param_search_round2.py --stage 5", "Wrote final conservative search comparison and recommendation.", [out, "reports/search_round2_comparison.md", "logs/stage_search2_5.log", "reports/checkpoint_search2_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Search2 stage 5 failed")
    print(smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", default="configs/pseudo_user_search_round2.yaml")
    args = parser.parse_args()
    stages = {1: stage_1, 2: stage_2, 3: stage_3, 4: stage_4, 5: stage_5}
    stages[args.stage](args.config)


if __name__ == "__main__":
    main()
