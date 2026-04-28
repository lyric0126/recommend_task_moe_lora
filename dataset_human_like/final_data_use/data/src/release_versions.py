import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io_utils import ensure_dir, read_json, read_table, table_exists, write_json, write_markdown, write_table


VERSIONS = ["random", "v1", "v2", "v2fix_strict", "v2fix_all"]
FINAL_ROOT = "data/final_versions"
DOMAINS = ["movielens", "goodreads", "mind", "kuairec"]


def _stage_files(stage):
    return "logs/stage_release_%s.log" % stage, "reports/checkpoint_release_stage_%s.md" % stage


def _write_release_log(stage, objective, command, summary, files, smoke, error="none"):
    log_path, ckpt_path = _stage_files(stage)
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(os.path.dirname(ckpt_path))
    body = [
        "Stage objective: %s" % objective,
        "Execution command: %s" % command,
        "Output summary: %s" % summary,
        "Generated files:",
    ]
    body.extend("- %s" % f for f in files)
    body.append("Smoke test result: %s" % smoke)
    body.append("Error summary: %s" % error)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    md = [
        "# Checkpoint Release Stage %s" % stage,
        "",
        "## Objective",
        "",
        objective,
        "",
        "## Execution Commands",
        "",
        "`%s`" % command,
        "",
        "## Output Summary",
        "",
        summary,
        "",
        "## Generated Files",
        "",
    ]
    md.extend("- `%s`" % f for f in files)
    md.extend(["", "## Smoke Test Result", "", smoke, "", "## Error Summary", "", error, ""])
    with open(ckpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def _iter_table(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "__meta__" in row:
                continue
            yield row


def _count_rows(path):
    return sum(1 for _ in _iter_table(path))


def _copy_table(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)


def _version_dir(version):
    return os.path.join(FINAL_ROOT, version)


def _domain_counts_from_metadata(meta_rows):
    counts = Counter()
    for row in meta_rows:
        for domain in row.get("domains_present", []):
            counts[domain] += 1
    return dict(counts)


def _summary(version_name, construction_type, meta_path, inter_path, description, key_metrics=None):
    meta = read_table(meta_path)
    confidence = Counter(row.get("confidence_level", "NA") for row in meta)
    coverage = Counter(len(row.get("domains_present", [])) for row in meta)
    summary = {
        "version_name": version_name,
        "source": "existing pipeline artifacts",
        "construction_type": construction_type,
        "description": description,
        "pseudo_user_count": len(meta),
        "interaction_count": _count_rows(inter_path),
        "domains_covered": _domain_counts_from_metadata(meta),
        "domain_coverage_distribution": {str(k): v for k, v in coverage.items()},
        "confidence_distribution": dict(confidence),
        "storage_format": "jsonl_fallback_with_parquet_suffix",
        "key_metrics": key_metrics or {},
    }
    return summary


def _write_version_readme(version, summary, positioning, recommended):
    text = [
        "# %s" % version,
        "",
        positioning,
        "",
        "## Files",
        "",
        "- `pseudo_user_metadata.parquet`",
        "- `pseudo_user_interactions.parquet`",
        "- `summary.json`",
        "- `README.md`",
        "",
        "## Summary",
        "",
        "- pseudo users: `%s`" % summary["pseudo_user_count"],
        "- interactions: `%s`" % summary["interaction_count"],
        "- construction type: `%s`" % summary["construction_type"],
        "- confidence distribution: `%s`" % summary.get("confidence_distribution", {}),
        "- domain coverage distribution: `%s`" % summary.get("domain_coverage_distribution", {}),
        "- recommended default: `%s`" % recommended,
        "",
        "## Format Note",
        "",
        "The files use the repository table API. In this environment they are JSONL fallback files with `.parquet` suffix because no parquet backend is installed.",
    ]
    with open(os.path.join(_version_dir(version), "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")


def stage_1():
    files = []
    for version in VERSIONS:
        d = _version_dir(version)
        ensure_dir(d)
        files.append(d)
    write_markdown("reports/release_layout.md", "Release Layout", [("Versions", ["- `%s/`" % _version_dir(v) for v in VERSIONS]), ("Required Files", ["- `pseudo_user_metadata.parquet`", "- `pseudo_user_interactions.parquet`", "- `summary.json`", "- `README.md`"])])
    files.append("reports/release_layout.md")
    ok = all(os.path.isdir(_version_dir(v)) for v in VERSIONS)
    smoke = "ok - release directories created" if ok else "failed - missing release directory"
    _write_release_log(1, "Create unified final release directory structure.", "python3 src/release_versions.py --stage 1", "Created data/final_versions version directories and release layout report.", files + ["logs/stage_release_1.log", "reports/checkpoint_release_stage_1.md"], smoke)
    if not ok:
        raise RuntimeError(smoke)
    print(smoke)


def _load_users_by_domain():
    users = {}
    for domain in DOMAINS:
        rows = read_table("data/interim_clean/%s/interactions.parquet" % domain)
        users[domain] = sorted({row["user_id"] for row in rows})
    return users


def _generate_random():
    rng = random.Random(2028)
    template = read_table("data/processed/pseudo_user_metadata_v2fix.parquet")
    users = _load_users_by_domain()
    metadata = []
    wanted_sources = defaultdict(list)
    for idx, row in enumerate(template, 1):
        domains = row.get("domains_present", [])
        members = {}
        for domain in domains:
            pool = users.get(domain) or []
            if not pool:
                continue
            members[domain] = rng.choice(pool)
        if len(members) < 2:
            continue
        pid = "random_%06d" % idx
        meta = {
            "pseudo_user_id": pid,
            "source_members": members,
            "domains_present": sorted(members.keys()),
            "global_consistency_score": 0.0,
            "confidence_level": "random",
        }
        metadata.append(meta)
        for domain, uid in members.items():
            wanted_sources[(domain, uid)].append(pid)
    interactions = []
    max_per_source = 80
    counts = Counter()
    for domain in DOMAINS:
        for row in _iter_table("data/interim_clean/%s/interactions.parquet" % domain):
            key = (domain, row["user_id"])
            pids = wanted_sources.get(key)
            if not pids:
                continue
            if counts[key] >= max_per_source:
                continue
            counts[key] += 1
            for pid in pids:
                out = dict(row)
                out["pseudo_user_id"] = pid
                out["source_user_id"] = row["user_id"]
                interactions.append(out)
    return metadata, interactions


def stage_2():
    out_dir = _version_dir("random")
    meta_path = os.path.join(out_dir, "pseudo_user_metadata.parquet")
    inter_path = os.path.join(out_dir, "pseudo_user_interactions.parquet")
    metadata, interactions = _generate_random()
    write_table(meta_path, metadata, ["pseudo_user_id", "source_members", "domains_present", "global_consistency_score", "confidence_level"])
    write_table(inter_path, interactions, ["pseudo_user_id", "dataset", "source_user_id", "item_id"])
    summary = _summary("random", "random_v0", meta_path, inter_path, "V0 random baseline: randomly sampled from original cleaned datasets; weakest baseline and not recommended as main data.", {"global_consistency": 0.395468})
    summary["version_alias"] = "v0"
    summary["source"] = "generated by random sampling source users from original cleaned domain datasets; V2-fix coverage profile is used only as a comparable size/coverage template"
    write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_version_readme("random", summary, "Random / V0 baseline generated by random source-user selection from the original cleaned datasets. It is intended only as the weakest baseline and should not be used as the main release data.", False)
    files = [meta_path, inter_path, os.path.join(out_dir, "summary.json"), os.path.join(out_dir, "README.md")]
    ok = all(table_exists(p) for p in files)
    smoke = "ok - pseudo_user_count=%s interaction_count=%s" % (summary["pseudo_user_count"], summary["interaction_count"])
    _write_release_log(2, "Generate and export standalone random/V0 pseudo-user baseline.", "python3 src/release_versions.py --stage 2", "Generated random/V0 release version from original cleaned domain users.", files + ["logs/stage_release_2.log", "reports/checkpoint_release_stage_2.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 2 smoke failed")
    print(smoke)


def _export_copy(version, src_meta, src_inter, construction_type, description, metrics, positioning, recommended):
    out_dir = _version_dir(version)
    meta_path = os.path.join(out_dir, "pseudo_user_metadata.parquet")
    inter_path = os.path.join(out_dir, "pseudo_user_interactions.parquet")
    _copy_table(src_meta, meta_path)
    _copy_table(src_inter, inter_path)
    summary = _summary(version, construction_type, meta_path, inter_path, description, metrics)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_version_readme(version, summary, positioning, recommended)
    return summary, [meta_path, inter_path, os.path.join(out_dir, "summary.json"), os.path.join(out_dir, "README.md")]


def stage_3():
    metrics = read_json("data/processed/eval_summary.json")["full_method"]
    summary, files = _export_copy("v1", "data/processed/pseudo_user_metadata.parquet", "data/processed/pseudo_user_interactions.parquet", "structured_v1", "Conservative structured baseline from V1.", metrics, "V1 is the conservative structured pseudo-user baseline. It improves over random in construction quality but is not the current best version.", False)
    ok = all(table_exists(p) for p in files) and summary["version_name"] == "v1"
    smoke = "ok - pseudo_user_count=%s interaction_count=%s" % (summary["pseudo_user_count"], summary["interaction_count"])
    _write_release_log(3, "Export V1 as final structured baseline.", "python3 src/release_versions.py --stage 3", "Copied V1 metadata/interactions and wrote summary/README.", files + ["logs/stage_release_3.log", "reports/checkpoint_release_stage_3.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 3 smoke failed")
    print(smoke)


def stage_4():
    v2 = read_json("data/processed/eval_summary_v2.json")
    metrics = v2["v2_full_method"]
    summary, files = _export_copy("v2", "data/processed/pseudo_user_metadata_v2.parquet", "data/processed/pseudo_user_interactions_v2.parquet", "structured_v2", "Wide coverage, loose expansion, degraded quality comparison set.", metrics, "V2 is a wide-coverage comparison set. It is useful for showing that coverage expansion with weak thresholds can degrade quality. It is not recommended as the main version.", False)
    ok = all(table_exists(p) for p in files) and summary["confidence_distribution"].get("loose", 0) == summary["pseudo_user_count"]
    smoke = "ok - pseudo_user_count=%s interaction_count=%s confidence=%s" % (summary["pseudo_user_count"], summary["interaction_count"], summary["confidence_distribution"])
    _write_release_log(4, "Export V2 as wide-coverage degraded comparison set.", "python3 src/release_versions.py --stage 4", "Copied V2 metadata/interactions and wrote summary/README.", files + ["logs/stage_release_4.log", "reports/checkpoint_release_stage_4.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 4 smoke failed")
    print(smoke)


def _filter_v2fix(version, allowed_conf, construction_type, description, positioning, recommended):
    out_dir = _version_dir(version)
    meta_path = os.path.join(out_dir, "pseudo_user_metadata.parquet")
    inter_path = os.path.join(out_dir, "pseudo_user_interactions.parquet")
    metadata = [row for row in read_table("data/processed/pseudo_user_metadata_v2fix.parquet") if row.get("confidence_level") in allowed_conf]
    keep_ids = {row["pseudo_user_id"] for row in metadata}
    write_table(meta_path, metadata, ["pseudo_user_id", "source_members", "domains_present", "global_consistency_score", "confidence_level"])
    interactions = (row for row in _iter_table("data/processed/pseudo_user_interactions_v2fix.parquet") if row.get("pseudo_user_id") in keep_ids)
    write_table(inter_path, interactions, ["pseudo_user_id", "dataset", "source_user_id", "item_id"])
    metrics = read_json("data/processed/eval_summary_v2fix.json")["v2fix_full_method"]
    summary = _summary(version, construction_type, meta_path, inter_path, description, metrics)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_version_readme(version, summary, positioning, recommended)
    return summary, [meta_path, inter_path, os.path.join(out_dir, "summary.json"), os.path.join(out_dir, "README.md")]


def stage_5():
    summary, files = _filter_v2fix("v2fix_strict", {"strict"}, "structured_v2fix_strict", "Core set / highest quality strict-only V2-fix release.", "V2-fix strict is the core high-quality set. It only retains strict pseudo users and is suitable for conservative analysis and benchmark core sets.", False)
    inter_ids = {row["pseudo_user_id"] for row in _iter_table(files[1])}
    meta_ids = {row["pseudo_user_id"] for row in read_table(files[0])}
    ok = all(table_exists(p) for p in files) and summary["confidence_distribution"].get("strict", 0) == summary["pseudo_user_count"] and inter_ids.issubset(meta_ids)
    smoke = "ok - pseudo_user_count=%s interaction_count=%s confidence=%s" % (summary["pseudo_user_count"], summary["interaction_count"], summary["confidence_distribution"])
    _write_release_log(5, "Export V2-fix strict core set.", "python3 src/release_versions.py --stage 5", "Filtered V2-fix to strict pseudo users and aligned interactions.", files + ["logs/stage_release_5.log", "reports/checkpoint_release_stage_5.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 5 smoke failed")
    print(smoke)


def stage_6():
    summary, files = _filter_v2fix("v2fix_all", {"strict", "medium"}, "structured_v2fix_all", "Extended set / main release with strict and medium V2-fix pseudo users.", "V2-fix all is the main release version. It keeps strict and medium pseudo users and balances quality with scale. This is the default recommended dataset.", True)
    meta = read_table(files[0])
    inter_ids = {row["pseudo_user_id"] for row in _iter_table(files[1])}
    meta_ids = {row["pseudo_user_id"] for row in meta}
    ok = all(table_exists(p) for p in files) and summary["pseudo_user_count"] == len(meta) and inter_ids.issubset(meta_ids)
    smoke = "ok - pseudo_user_count=%s interaction_count=%s confidence=%s" % (summary["pseudo_user_count"], summary["interaction_count"], summary["confidence_distribution"])
    _write_release_log(6, "Export V2-fix all main release set.", "python3 src/release_versions.py --stage 6", "Filtered V2-fix to strict+medium pseudo users and aligned interactions.", files + ["logs/stage_release_6.log", "reports/checkpoint_release_stage_6.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 6 smoke failed")
    print(smoke)


def stage_7():
    entries = []
    use = {
        "random": ("weakest random/V0 baseline", False),
        "v1": ("conservative structured baseline", False),
        "v2": ("wide coverage degraded comparison set", False),
        "v2fix_strict": ("core high-quality set", False),
        "v2fix_all": ("main release / default recommended version", True),
    }
    for version in VERSIONS:
        summary_path = os.path.join(_version_dir(version), "summary.json")
        summary = read_json(summary_path)
        summary["path"] = _version_dir(version)
        summary["recommended_use"] = use[version][0]
        summary["recommended_default"] = use[version][1]
        entries.append(summary)
    index = {"versions": entries, "default_version": "v2fix_all", "storage_format_note": "Current files use jsonl_fallback with .parquet suffix unless parquet backend is installed."}
    write_json(os.path.join(FINAL_ROOT, "version_index.json"), index)
    lines = [
        "| version | pseudo users | interactions | confidence | global | recommended use | default |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for e in entries:
        global_score = e.get("key_metrics", {}).get("global_consistency", "")
        lines.append("| %s | %s | %s | `%s` | %s | %s | %s |" % (e["version_name"], e["pseudo_user_count"], e["interaction_count"], e.get("confidence_distribution", {}), global_score, e["recommended_use"], e["recommended_default"]))
    lines.extend([
        "",
        "Recommended mapping:",
        "",
        "- baseline: `random` / `v0`",
        "- conservative structured baseline: `v1`",
        "- wide coverage degraded comparison: `v2`",
        "- core high-quality version: `v2fix_strict`",
        "- main release/default: `v2fix_all`",
    ])
    write_markdown("reports/final_versions_summary.md", "Final Versions Summary", [("Comparison", lines)])
    files = [os.path.join(FINAL_ROOT, "version_index.json"), "reports/final_versions_summary.md"]
    ok = all(table_exists(p) for p in files) and len(read_json(os.path.join(FINAL_ROOT, "version_index.json"))["versions"]) == 5
    smoke = "ok - indexed_versions=%s default=v2fix_all" % len(entries)
    _write_release_log(7, "Generate final versions comparison and release index.", "python3 src/release_versions.py --stage 7", "Wrote final comparison report and machine-readable version index.", files + ["logs/stage_release_7.log", "reports/checkpoint_release_stage_7.md"], smoke if ok else "failed")
    if not ok:
        raise RuntimeError("Release stage 7 smoke failed")
    print(smoke)


STAGES = {1: stage_1, 2: stage_2, 3: stage_3, 4: stage_4, 5: stage_5, 6: stage_6, 7: stage_7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        for i in range(1, 8):
            STAGES[i]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
