import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io_utils import load_config, write_stage_log


def stage_1(args):
    config = load_config(args.config)
    datasets = ", ".join(config.get("datasets", []))
    files = [
        "configs/pseudo_user_pipeline.yaml",
        "src/schema.py",
        "src/io_utils.py",
        "src/run_pipeline.py",
    ]
    smoke = "ok - config read successfully; datasets=%s" % datasets
    print(smoke)
    write_stage_log(
        1,
        "Create runnable pipeline skeleton and verify config loading.",
        "python3 src/run_pipeline.py --stage 1 --config %s" % args.config,
        "Pipeline skeleton exists and config loads successfully.",
        files + ["logs/stage_1.log", "reports/checkpoint_stage_1.md"],
        smoke,
    )


def stage_2(args):
    from src.io_utils import table_exists, write_markdown
    from src.loaders.datasets import run_loaders

    config = load_config(args.config)
    results = run_loaders(config)
    lines = []
    files = []
    for result in results:
        files.extend(result["paths"])
        lines.append("- %s: interactions=%s items=%s columns=%s" % (result["dataset"], result["interactions"], result["items"], result["columns"]))
        lines.append("  sample: `%s`" % (result["sample"],))
    write_markdown("reports/stage_2_loader_summary.md", "Stage 2 Loader Summary", [("Loader Outputs", lines)])
    files.append("reports/stage_2_loader_summary.md")
    ok = all(table_exists(path) for path in files if path.endswith(".parquet"))
    smoke = "ok - loader parquet fallback files exist" if ok else "failed - missing loader output"
    print(smoke)
    print("\n".join(lines[:12]))
    write_stage_log(2, "Implement four dataset loaders and write interim interactions/items.", "python3 src/run_pipeline.py --stage 2 --config %s" % args.config, "Loaded MovieLens, Goodreads, MIND, and KuaiRec into canonical schema.", files + ["logs/stage_2.log", "reports/checkpoint_stage_2.md"], smoke)
    if not ok:
        raise RuntimeError(smoke)


def stage_3(args):
    from src.io_utils import table_exists
    from src.normalize.cleaning import run_cleaning

    config = load_config(args.config)
    results = run_cleaning(config)
    files = ["reports/stage_3_cleaning_summary.md"]
    for r in results:
        files.extend(r["paths"])
    ok = all(table_exists(path) for path in files if path.endswith(".parquet"))
    smoke = "ok - cleaned outputs exist; " + "; ".join("%s %s->%s" % (r["dataset"], r["before_interactions"], r["after_interactions"]) for r in results)
    print(smoke)
    write_stage_log(3, "Clean and standardize all four domains.", "python3 src/run_pipeline.py --stage 3 --config %s" % args.config, "Applied min interaction filters, timestamp normalization, text/category standardization.", files + ["logs/stage_3.log", "reports/checkpoint_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 3 smoke failed")


def stage_4(args):
    from src.io_utils import table_exists
    from src.features.embeddings import build_item_embeddings

    config = load_config(args.config)
    result = build_item_embeddings(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - item embeddings rows=%s dim=%s" % (result["rows"], result["dim"])
    print(smoke)
    write_stage_log(4, "Build item embeddings with deterministic fallback.", "python3 src/run_pipeline.py --stage 4 --config %s" % args.config, "Generated item embeddings for all cleaned domains using deterministic hashing fallback.", [result["path"], "reports/stage_4_embedding_summary.md", "logs/stage_4.log", "reports/checkpoint_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 4 smoke failed")


def stage_5(args):
    from src.io_utils import table_exists
    from src.features.profiles import build_user_profiles

    config = load_config(args.config)
    result = build_user_profiles(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - user profiles rows=%s" % result["rows"]
    print(smoke)
    write_stage_log(5, "Build semantic/activity/temporal/behavior user profiles.", "python3 src/run_pipeline.py --stage 5 --config %s" % args.config, "Generated user profile records for each cleaned domain.", [result["path"], "reports/stage_5_user_profiles_summary.md", "logs/stage_5.log", "reports/checkpoint_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 5 smoke failed")


def stage_6(args):
    from src.io_utils import table_exists
    from src.matching.matcher import run_matching

    config = load_config(args.config)
    results = run_matching(config)
    files = [r["path"] for r in results] + ["reports/stage_6_matching_summary.md"]
    ok = all(table_exists(p) for p in files if p.endswith(".parquet"))
    smoke = "ok - matches generated: " + "; ".join("%s rows=%s" % (r["pair"], r["rows"]) for r in results)
    print(smoke)
    write_stage_log(6, "Run two-domain matching for MovieLens-Goodreads and MIND-KuaiRec.", "python3 src/run_pipeline.py --stage 6 --config %s" % args.config, "Generated blocked candidate matches and weighted scores.", files + ["logs/stage_6.log", "reports/checkpoint_stage_6.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 6 smoke failed")


def stage_7(args):
    from src.io_utils import table_exists
    from src.synthesis.pseudo_users import synthesize

    config = load_config(args.config)
    result = synthesize(config)
    ok = all(table_exists(p) for p in result["paths"]) and result["metadata"] > 0
    smoke = "ok - pseudo_users=%s interactions=%s coverage=%s confidence=%s" % (result["metadata"], result["interactions"], result["coverage"], result["confidence"])
    print(smoke)
    write_stage_log(7, "Synthesize MovieLens-anchored pseudo users across Goodreads, MIND, and KuaiRec.", "python3 src/run_pipeline.py --stage 7 --config %s" % args.config, "Generated pseudo-user metadata and merged pseudo-user interactions.", result["paths"] + ["reports/stage_7_synthesis_summary.md", "logs/stage_7.log", "reports/checkpoint_stage_7.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 7 smoke failed")


def stage_8(args):
    from src.evaluation.evaluate import evaluate
    from src.io_utils import table_exists

    config = load_config(args.config)
    result = evaluate(config)
    ok = table_exists(result["path"]) and table_exists("reports/pseudo_user_eval.md")
    smoke = "ok - evaluation complete Random Mix vs Full Method: random=%s full=%s" % (result["summary"]["random_mix"], result["summary"]["full_method"])
    print(smoke)
    write_stage_log(8, "Run minimal evaluation with Random Mix baseline.", "python3 src/run_pipeline.py --stage 8 --config %s" % args.config, "Wrote evaluation summary and markdown report.", [result["path"], "reports/pseudo_user_eval.md", "logs/stage_8.log", "reports/checkpoint_stage_8.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Stage 8 smoke failed")


STAGES = {
    1: stage_1,
    2: stage_2,
    3: stage_3,
    4: stage_4,
    5: stage_5,
    6: stage_6,
    7: stage_7,
    8: stage_8,
}


def v2_stage_1(args):
    from src.io_utils import read_table, table_storage_info, write_markdown, write_table, write_v2_stage_log

    config = load_config(args.config)
    sample_path = "data/processed/v2_storage_smoke.parquet"
    rows = [{"dataset": "smoke", "user_id": "u1", "item_id": "i1", "score": 1.0}]
    write_table(sample_path, rows, ["dataset", "user_id", "item_id", "score"], config["pipeline"].get("storage_format", "auto"))
    back = read_table(sample_path)
    info = table_storage_info(sample_path)
    lines = ["storage_info=`%s`" % info, "rows_read=%s" % len(back), "sample=`%s`" % back[:1]]
    write_markdown("reports/v2_storage_upgrade.md", "V2 Storage Upgrade", [("Smoke Test", lines)])
    files = [sample_path, "src/io_utils.py", "reports/v2_storage_upgrade.md", "configs/pseudo_user_pipeline_v2.yaml"]
    smoke = "ok - storage=%s rows=%s schema=%s" % (info["storage_format"], len(back), info["schema"])
    print(smoke)
    write_v2_stage_log(1, "Upgrade storage layer with parquet auto-detection and JSONL fallback metadata.", "python3 src/run_pipeline.py --v2-stage 1 --config %s" % args.config, "Storage smoke wrote and read a sample table.", files + ["logs/stage_v2_1.log", "reports/checkpoint_v2_stage_1.md"], smoke)


def v2_stage_2(args):
    from src.features.embeddings_v2 import build_item_embeddings_v2
    from src.io_utils import table_exists, write_v2_stage_log

    config = load_config(args.config)
    result = build_item_embeddings_v2(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - backend=%s rows=%s dim=%s coverage=%s" % (result["backend"], result["rows"], result["dim"], result["coverage"])
    print(smoke)
    write_v2_stage_log(2, "Upgrade item text representation and embedding backend.", "python3 src/run_pipeline.py --v2-stage 2 --config %s" % args.config, "Generated V2 item embeddings with enhanced text and configurable fallback backend.", [result["path"], "reports/v2_item_representation.md", "logs/stage_v2_2.log", "reports/checkpoint_v2_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2 stage 2 smoke failed")


def v2_stage_3(args):
    from src.features.profiles_v2 import build_user_profiles_v2
    from src.io_utils import table_exists, write_v2_stage_log

    config = load_config(args.config)
    result = build_user_profiles_v2(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - v2 user profiles rows=%s" % result["rows"]
    print(smoke)
    write_v2_stage_log(3, "Upgrade user profiles with recency semantics, diversity, structured fields, and better sessions.", "python3 src/run_pipeline.py --v2-stage 3 --config %s" % args.config, "Generated V2 user profiles and compared field coverage against V1.", [result["path"], "reports/v2_user_profiles.md", "logs/stage_v2_3.log", "reports/checkpoint_v2_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2 stage 3 smoke failed")


def v2_stage_4(args):
    from src.io_utils import table_exists, write_v2_stage_log
    from src.matching.matcher_v2 import run_matching_v2

    config = load_config(args.config)
    results = run_matching_v2(config)
    files = [r["path"] for r in results] + ["reports/v2_matching_upgrade.md"]
    ok = all(table_exists(p) for p in files if p.endswith(".parquet"))
    smoke = "ok - " + "; ".join("%s rows=%s confidence=%s" % (r["pair"], r["rows"], r["confidence"]) for r in results)
    print(smoke)
    write_v2_stage_log(4, "Upgrade matching with stronger blocking, configurable scoring, thresholds, and ablations.", "python3 src/run_pipeline.py --v2-stage 4 --config %s" % args.config, "Generated V2 matches and ablation scores.", files + ["logs/stage_v2_4.log", "reports/checkpoint_v2_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2 stage 4 smoke failed")


def v2_stage_5(args):
    from src.io_utils import table_exists, write_v2_stage_log
    from src.synthesis.pseudo_users_v2 import synthesize_v2

    config = load_config(args.config)
    result = synthesize_v2(config)
    ok = all(table_exists(p) for p in result["paths"]) and result["metadata"] > 0
    smoke = "ok - pseudo_users=%s interactions=%s coverage=%s confidence=%s" % (result["metadata"], result["interactions"], result["coverage"], result["confidence"])
    print(smoke)
    write_v2_stage_log(5, "Upgrade pseudo-user synthesis with 2/3/4-domain support and improved global consistency.", "python3 src/run_pipeline.py --v2-stage 5 --config %s" % args.config, "Generated V2 pseudo-user metadata and interactions.", result["paths"] + ["reports/v2_synthesis_upgrade.md", "logs/stage_v2_5.log", "reports/checkpoint_v2_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2 stage 5 smoke failed")


def v2_stage_6(args):
    from src.evaluation.evaluate_v2 import evaluate_v2
    from src.io_utils import table_exists, write_v2_stage_log

    config = load_config(args.config)
    result = evaluate_v2(config)
    ok = table_exists(result["path"]) and table_exists("reports/pseudo_user_eval_v2.md")
    smoke = "ok - V1 global=%s V2 global=%s V2_better=%s" % (result["summary"]["v1"].get("full_method", {}).get("global_consistency"), result["summary"]["v2_full_method"].get("global_consistency"), result["summary"]["v2_better_than_v1_global"])
    print(smoke)
    write_v2_stage_log(6, "Upgrade evaluation with V1/V2 comparison, ablations, confidence metrics, and coverage/reuse reporting.", "python3 src/run_pipeline.py --v2-stage 6 --config %s" % args.config, "Generated V2 evaluation summary and report.", [result["path"], "reports/pseudo_user_eval_v2.md", "logs/stage_v2_6.log", "reports/checkpoint_v2_stage_6.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2 stage 6 smoke failed")


V2_STAGES = {
    1: v2_stage_1,
    2: v2_stage_2,
    3: v2_stage_3,
    4: v2_stage_4,
    5: v2_stage_5,
    6: v2_stage_6,
}


def v2fix_stage_1(args):
    from src.features.embeddings_v2fix import build_item_embeddings_v2fix
    from src.io_utils import table_exists, write_v2fix_stage_log

    config = load_config(args.config)
    result = build_item_embeddings_v2fix(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - rows=%s dim=%s coverage=%s" % (result["rows"], result["dim"], result["coverage"])
    print(smoke)
    write_v2fix_stage_log(1, "Strengthen semantic item representation with cleaner text and stable TF-IDF hashing fallback.", "python3 src/run_pipeline.py --v2fix-stage 1 --config %s" % args.config, "Generated V2-fix item embeddings and similarity diagnostics.", [result["path"], "reports/v2fix_embeddings.md", "logs/stage_v2fix_1.log", "reports/checkpoint_v2fix_stage_1.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2fix stage 1 smoke failed")


def v2fix_stage_2(args):
    from src.features.profiles_v2fix import build_user_profiles_v2fix
    from src.io_utils import table_exists, write_v2fix_stage_log

    config = load_config(args.config)
    result = build_user_profiles_v2fix(config)
    ok = table_exists(result["path"]) and result["rows"] > 0
    smoke = "ok - rows=%s" % result["rows"]
    print(smoke)
    write_v2fix_stage_log(2, "Rebuild user semantic profiles from fixed embeddings while preserving structured activity/temporal/behavior fields.", "python3 src/run_pipeline.py --v2fix-stage 2 --config %s" % args.config, "Generated V2-fix user profiles and profile samples.", [result["path"], "reports/v2fix_user_profiles.md", "logs/stage_v2fix_2.log", "reports/checkpoint_v2fix_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2fix stage 2 smoke failed")


def v2fix_stage_3(args):
    from src.io_utils import table_exists, write_v2fix_stage_log
    from src.matching.matcher_v2fix import run_matching_v2fix

    config = load_config(args.config)
    results = run_matching_v2fix(config)
    files = [r["path"] for r in results] + ["reports/v2fix_matching.md"]
    ok = all(table_exists(p) for p in files if p.endswith(".parquet"))
    collapsed = all((r["confidence"].get("strict", 0) + r["confidence"].get("medium", 0)) == 0 for r in results)
    smoke = "ok - " + "; ".join("%s rows=%s confidence=%s" % (r["pair"], r["rows"], r["confidence"]) for r in results)
    if collapsed:
        smoke = "failed - confidence still collapsed to loose"
    print(smoke)
    write_v2fix_stage_log(3, "Tighten matching strategy with stricter blocking, smaller fallback candidates, calibrated scoring, and diagnostics.", "python3 src/run_pipeline.py --v2fix-stage 3 --config %s" % args.config, "Generated V2-fix matches and score diagnostics.", files + ["logs/stage_v2fix_3.log", "reports/checkpoint_v2fix_stage_3.md"], smoke if ok and not collapsed else "failed", "" if ok and not collapsed else "strict/medium counts are zero across all pairs")
    if not ok or collapsed:
        raise RuntimeError("V2fix stage 3 smoke failed")


def v2fix_stage_4(args):
    from src.io_utils import table_exists, write_v2fix_stage_log
    from src.synthesis.pseudo_users_v2fix import synthesize_v2fix

    config = load_config(args.config)
    result = synthesize_v2fix(config)
    ok = all(table_exists(p) for p in result["paths"]) and result["metadata"] > 0
    smoke = "ok - pseudo_users=%s interactions=%s coverage=%s confidence=%s" % (result["metadata"], result["interactions"], result["coverage"], result["confidence"])
    print(smoke)
    write_v2fix_stage_log(4, "Redo pseudo-user synthesis with medium+ component filtering, consistency filtering, and interaction caps.", "python3 src/run_pipeline.py --v2fix-stage 4 --config %s" % args.config, "Generated V2-fix pseudo-user metadata and interactions.", result["paths"] + ["reports/v2fix_synthesis.md", "logs/stage_v2fix_4.log", "reports/checkpoint_v2fix_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2fix stage 4 smoke failed")


def v2fix_stage_5(args):
    from src.evaluation.evaluate_v2fix import evaluate_v2fix
    from src.io_utils import table_exists, write_v2fix_stage_log

    config = load_config(args.config)
    result = evaluate_v2fix(config)
    ok = table_exists(result["path"]) and table_exists("reports/pseudo_user_eval_v2fix.md")
    s = result["summary"]
    smoke = "ok - V1=%s V2=%s V2fix=%s better_than_v2=%s close_to_v1=%s full_gt_semantic=%s full_minus_activity_temporal=%s" % (
        s["v1_global"], s["v2_global"], s["v2fix_full_method"].get("global_consistency"), s["v2fix_better_than_v2"], s["v2fix_close_to_or_above_v1"], s["full_better_than_semantic_only"], s["full_vs_activity_temporal_gap"]
    )
    print(smoke)
    write_v2fix_stage_log(5, "Evaluate V2-fix against Random Mix, V1, and V2 with ablations and confidence metrics.", "python3 src/run_pipeline.py --v2fix-stage 5 --config %s" % args.config, "Generated V2-fix evaluation summary and comparison report.", [result["path"], "reports/pseudo_user_eval_v2fix.md", "logs/stage_v2fix_5.log", "reports/checkpoint_v2fix_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V2fix stage 5 smoke failed")


V2FIX_STAGES = {
    1: v2fix_stage_1,
    2: v2fix_stage_2,
    3: v2fix_stage_3,
    4: v2fix_stage_4,
    5: v2fix_stage_5,
}


def v3_stage_1(args):
    from src.experiments.splits import build_splits
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = build_splits(config)
    ok = all(table_exists(p) for p in result["files"])
    smoke = "ok - " + " | ".join(result["summary"][:4])
    print(smoke)
    write_v3_stage_log(1, "Build fixed leave-last-2-out train/val/test splits for downstream target recommendation.", "python3 src/run_pipeline.py --v3-stage 1 --config %s" % args.config, "Generated MovieLens and Goodreads experimental splits.", result["files"] + ["logs/stage_v3_1.log", "reports/checkpoint_v3_stage_1.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 1 smoke failed")


def v3_stage_2(args):
    from src.experiments.training_sets import build_training_sets
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = build_training_sets(config)
    ok = all(table_exists(p) for p in result["files"])
    smoke = "ok - built %s training set files" % (len([p for p in result["files"] if p.endswith(".parquet")]))
    print(smoke)
    write_v3_stage_log(2, "Construct single-domain, random-mix, and pseudo-user(V2-fix) training sets.", "python3 src/run_pipeline.py --v3-stage 2 --config %s" % args.config, "Generated downstream training sets with leakage filtering.", result["files"] + ["logs/stage_v3_2.log", "reports/checkpoint_v3_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 2 smoke failed")


def v3_stage_3(args):
    from src.experiments.baselines import run_baseline_smoke
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = run_baseline_smoke(config)
    ok = table_exists(result["path"]) and table_exists("reports/v3_baselines.md")
    smoke = "ok - " + " | ".join(result["summary"])
    print(smoke)
    write_v3_stage_log(3, "Implement and smoke-test lightweight popularity and item-item baselines.", "python3 src/run_pipeline.py --v3-stage 3 --config %s" % args.config, "Baseline smoke produced top-k recommendations for one target user.", [result["path"], "reports/v3_baselines.md", "logs/stage_v3_3.log", "reports/checkpoint_v3_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 3 smoke failed")


def v3_stage_4(args):
    from src.experiments.main_experiment import run_main_experiment
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = run_main_experiment(config)
    ok = table_exists(result["path"]) and table_exists("reports/v3_main_results.md")
    checks = []
    for target in config["experiment"]["target_domains"]:
        pu = result["results"][target]["pseudo_user_v2fix"]["item_item"]["recall"]
        rm = result["results"][target]["random_mix"]["item_item"]["recall"]
        sd = result["results"][target]["single_domain"]["item_item"]["recall"]
        checks.append("%s pseudo>=random:%s pseudo>=single:%s" % (target, pu >= rm, pu >= sd))
    smoke = "ok - " + "; ".join(checks)
    print(smoke)
    write_v3_stage_log(4, "Run main downstream experiments comparing single-domain, random-mix, and pseudo-user(V2-fix).", "python3 src/run_pipeline.py --v3-stage 4 --config %s" % args.config, "Computed Recall/NDCG/HitRate/MRR for MovieLens and Goodreads.", [result["path"], "reports/v3_main_results.md", "logs/stage_v3_4.log", "reports/checkpoint_v3_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 4 smoke failed")


def v3_stage_5(args):
    from src.experiments.ablation import run_ablation
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = run_ablation(config)
    ok = table_exists(result["path"]) and table_exists("reports/v3_ablation.md")
    best = {}
    for target, by_variant in result["results"].items():
        best[target] = max(by_variant.items(), key=lambda kv: kv[1]["recall"])[0]
    smoke = "ok - best_by_recall=%s" % best
    print(smoke)
    write_v3_stage_log(5, "Run data-construction ablation across single-domain, random-mix, V1, V2, and V2-fix pseudo users.", "python3 src/run_pipeline.py --v3-stage 5 --config %s" % args.config, "Computed ablation metrics with item-item baseline.", [result["path"], "reports/v3_ablation.md", "logs/stage_v3_5.log", "reports/checkpoint_v3_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 5 smoke failed")


def v3_stage_6(args):
    from src.experiments.final_summary import write_final_summary
    from src.io_utils import table_exists, write_v3_stage_log

    config = load_config(args.config)
    result = write_final_summary(config)
    files = [result["path"], "data/processed/exp_results_v3.json", "data/processed/exp_ablation_v3.json"]
    ok = all(table_exists(p) for p in files)
    smoke = "ok - final report exists; answers=%s best_by_target=%s" % (result["answers"], result["best_by_target"])
    print(smoke)
    write_v3_stage_log(6, "Write final paper-style V3 experimental summary.", "python3 src/run_pipeline.py --v3-stage 6 --config %s" % args.config, "Generated final V3 summary with task, data, baselines, results, ablation, conclusions, limitations, and next steps.", files + ["logs/stage_v3_6.log", "reports/checkpoint_v3_stage_6.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("V3 stage 6 smoke failed")


V3_STAGES = {
    1: v3_stage_1,
    2: v3_stage_2,
    3: v3_stage_3,
    4: v3_stage_4,
    5: v3_stage_5,
    6: v3_stage_6,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pseudo-user synthesis pipeline")
    parser.add_argument("--config", default="configs/pseudo_user_pipeline.yaml")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--v2-stage", type=int, default=None)
    parser.add_argument("--v2fix-stage", type=int, default=None)
    parser.add_argument("--v3-stage", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="run stages 1 through 8")
    parser.add_argument("--v2-all", action="store_true", help="run V2 stages 1 through 6")
    parser.add_argument("--v2fix-all", action="store_true", help="run V2fix stages 1 through 5")
    parser.add_argument("--v3-all", action="store_true", help="run V3 stages 1 through 6")
    args = parser.parse_args(argv)
    if args.v3_all:
        for stage in range(1, 7):
            args.v3_stage = stage
            V3_STAGES[stage](args)
    elif args.v3_stage is not None:
        V3_STAGES[args.v3_stage](args)
    elif args.v2fix_all:
        for stage in range(1, 6):
            args.v2fix_stage = stage
            V2FIX_STAGES[stage](args)
    elif args.v2fix_stage is not None:
        V2FIX_STAGES[args.v2fix_stage](args)
    elif args.v2_all:
        for stage in range(1, 7):
            args.v2_stage = stage
            V2_STAGES[stage](args)
    elif args.v2_stage is not None:
        V2_STAGES[args.v2_stage](args)
    elif args.all:
        for stage in range(1, 9):
            args.stage = stage
            STAGES[stage](args)
    elif args.stage in STAGES:
        STAGES[args.stage](args)
    else:
        raise RuntimeError("Unknown stage %s" % args.stage)


if __name__ == "__main__":
    main()
