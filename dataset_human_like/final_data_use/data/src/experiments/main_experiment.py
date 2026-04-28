import os

from src.experiments.baselines import evaluate_model, train_model
from src.io_utils import read_json, read_table, write_json, write_markdown


def run_main_experiment(config):
    results = {}
    lines = []
    k = int(config["experiment"]["k_values"][0])
    for target in config["experiment"]["target_domains"]:
        split_dir = os.path.join(config["paths"]["exp_splits"], target)
        history_rows = read_table(os.path.join(split_dir, "train.parquet"))
        test_rows = read_table(os.path.join(split_dir, "test.parquet"))
        results[target] = {}
        for variant in config["experiment"]["training"]["variants"]:
            train_rows = read_table(os.path.join(config["paths"]["exp_sets"], target, "%s.parquet" % variant))
            results[target][variant] = {}
            for model_name in config["experiment"]["baseline"]["models"]:
                model = train_model(model_name, train_rows, config)
                metrics = evaluate_model(model, history_rows, test_rows, k)
                results[target][variant][model_name] = metrics
                lines.append("- %s/%s/%s Recall@%s=%s NDCG@%s=%s HitRate@%s=%s MRR@%s=%s users=%s" % (
                    target, variant, model_name, k, metrics["recall"], k, metrics["ndcg"], k, metrics["hitrate"], k, metrics["mrr"], metrics["users"]
                ))
        best_model = "item_item"
        pu = results[target]["pseudo_user_v2fix"][best_model]["recall"]
        rm = results[target]["random_mix"][best_model]["recall"]
        sd = results[target]["single_domain"][best_model]["recall"]
        lines.append("  pseudo_user_v2fix_vs_random_mix_%s=%s" % (best_model, pu >= rm))
        lines.append("  pseudo_user_v2fix_vs_single_domain_%s=%s" % (best_model, pu >= sd))
    out = os.path.join(config["paths"]["processed"], "exp_results_v3.json")
    write_json(out, results)
    write_markdown("reports/v3_main_results.md", "V3 Main Results", [("Smoke Test", lines)])
    return {"path": out, "results": results, "summary": lines}


def load_main_results(config):
    return read_json(os.path.join(config["paths"]["processed"], "exp_results_v3.json"))
