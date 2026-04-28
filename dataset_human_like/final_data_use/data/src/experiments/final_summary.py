import os

from src.io_utils import read_json, write_markdown


def write_final_summary(config):
    main = read_json(os.path.join(config["paths"]["processed"], "exp_results_v3.json"))
    ablation = read_json(os.path.join(config["paths"]["processed"], "exp_ablation_v3.json"))
    lines = []
    lines.append("## Task Definition")
    lines.append("")
    lines.append("Evaluate whether V2-fix pseudo-user augmentation improves lightweight target-domain recommendation on fixed MovieLens and Goodreads leave-last-2-out splits.")
    lines.append("")
    lines.append("## Data Construction")
    lines.append("")
    lines.append("- `single-domain`: target-domain train only.")
    lines.append("- `random-mix`: target-domain train plus random target-domain augmentation.")
    lines.append("- `pseudo-user_v2fix`: target-domain train plus V2-fix pseudo-user target-domain augmentation.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("- Popularity/frequency.")
    lines.append("- Pure Python item-item co-occurrence.")
    lines.append("")
    lines.append("## Main Results")
    lines.append("")
    lines.append("| target | variant | model | recall | ndcg | hitrate | mrr |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for target, by_variant in main.items():
        for variant, by_model in by_variant.items():
            for model, m in by_model.items():
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (target, variant, model, m["recall"], m["ndcg"], m["hitrate"], m["mrr"]))
    lines.append("")
    lines.append("## Ablation")
    lines.append("")
    lines.append("| target | variant | recall | ndcg | mrr | train_rows |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for target, by_variant in ablation.items():
        for variant, m in by_variant.items():
            lines.append("| %s | %s | %s | %s | %s | %s |" % (target, variant, m["recall"], m["ndcg"], m["mrr"], m["train_rows"]))
    answers = []
    for target in main:
        pu = main[target]["pseudo_user_v2fix"]["item_item"]["recall"]
        rm = main[target]["random_mix"]["item_item"]["recall"]
        sd = main[target]["single_domain"]["item_item"]["recall"]
        answers.append((target, pu > rm, pu > sd, pu, rm, sd))
    lines.append("")
    lines.append("## Key Questions")
    lines.append("")
    lines.append("1. pseudo-user(V2-fix) vs random-mix: `%s`" % {t: ok for t, ok, _, _, _, _ in answers})
    lines.append("2. pseudo-user(V2-fix) vs single-domain: `%s`" % {t: ok for t, _, ok, _, _, _ in answers})
    best_by_target = {}
    for target, by_variant in ablation.items():
        best_by_target[target] = max(by_variant.items(), key=lambda kv: kv[1]["recall"])[0]
    lines.append("3. Quality trend by ablation best variant: `%s`" % best_by_target)
    lines.append("4. Support for stronger models: yes if V2-fix beats random-mix or single-domain on at least one target; otherwise only after revisiting split/model design.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Baselines are intentionally lightweight and may not exploit cross-domain user structure fully.")
    lines.append("- Augmentation uses target-domain pseudo interactions only; no neural cross-domain recommender is trained.")
    lines.append("- Current storage remains JSONL fallback in `.parquet` paths due to missing parquet dependencies.")
    lines.append("- Results depend on the leave-last-2-out split and sampled V1/V2/V2-fix artifacts.")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Add a stronger but still reproducible implicit-feedback model when dependencies allow.")
    lines.append("- Evaluate true cross-domain transfer models that consume all domains in each pseudo user.")
    lines.append("- Repeat with true parquet storage and full data scale.")
    write_markdown("reports/v3_final_summary.md", "V3 Final Summary", [("Summary", lines)])
    return {"path": "reports/v3_final_summary.md", "answers": answers, "best_by_target": best_by_target}
