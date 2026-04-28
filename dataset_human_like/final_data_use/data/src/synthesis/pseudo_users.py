import os
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table
from src.matching.matcher import top_matches


def _confidence(score):
    if score >= 0.72:
        return "strict"
    if score >= 0.55:
        return "medium"
    return "loose"


def _by_dataset(profiles):
    out = defaultdict(list)
    for p in profiles:
        out[p["dataset"]].append(p)
    return out


def _top_by_left(matches):
    out = defaultdict(list)
    for m in matches:
        out[m["left_user_id"]].append(m)
    return out


def synthesize(config):
    profiles = read_table(os.path.join(config["paths"]["processed"], "user_profiles.parquet"))
    by_ds = _by_dataset(profiles)
    top_k = int(config["pipeline"].get("top_k_matches", 5))
    ml = by_ds["movielens"]
    gr_matches = read_table(os.path.join(config["paths"]["processed"], "matches_movielens_goodreads.parquet"))
    mind_matches, _ = top_matches(ml, by_ds["mind"], top_k)
    kr_matches, _ = top_matches(ml, by_ds["kuairec"], top_k)
    groups = {
        "goodreads": _top_by_left(gr_matches),
        "mind": _top_by_left(mind_matches),
        "kuairec": _top_by_left(kr_matches),
    }
    metadata = []
    pseudo_interactions = []
    for idx, anchor in enumerate(ml, 1):
        source_members = {"movielens": anchor["user_id"]}
        scores = []
        for domain in ["goodreads", "mind", "kuairec"]:
            match_list = groups[domain].get(anchor["user_id"], [])
            if match_list:
                source_members[domain] = match_list[0]["right_user_id"]
                scores.append(float(match_list[0]["score"]))
        if not scores:
            continue
        consistency = round(sum(scores) / len(scores), 6)
        domains_present = sorted(source_members.keys())
        pseudo_user_id = "pseudo_%06d" % idx
        metadata.append({
            "pseudo_user_id": pseudo_user_id,
            "source_members": source_members,
            "domains_present": domains_present,
            "global_consistency_score": consistency,
            "confidence_level": _confidence(consistency),
        })
    source_lookup = {}
    for meta in metadata:
        for ds, uid in meta["source_members"].items():
            source_lookup[(ds, uid)] = meta["pseudo_user_id"]
    for dataset in ["movielens", "goodreads", "mind", "kuairec"]:
        path = os.path.join(config["paths"]["interim_clean"], dataset, "interactions.parquet")
        for row in read_table(path):
            pid = source_lookup.get((dataset, row["user_id"]))
            if not pid:
                continue
            new_row = dict(row)
            new_row["pseudo_user_id"] = pid
            new_row["source_user_id"] = row["user_id"]
            pseudo_interactions.append(new_row)
    meta_path = os.path.join(config["paths"]["processed"], "pseudo_user_metadata.parquet")
    inter_path = os.path.join(config["paths"]["processed"], "pseudo_user_interactions.parquet")
    write_table(meta_path, metadata, ["pseudo_user_id", "source_members", "domains_present", "global_consistency_score", "confidence_level"])
    write_table(inter_path, pseudo_interactions, ["pseudo_user_id", "dataset", "source_user_id", "item_id"])
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    conf = Counter(m["confidence_level"] for m in metadata)
    lines = [
        "pseudo_users=%s" % len(metadata),
        "pseudo_interactions=%s" % len(pseudo_interactions),
        "domain_coverage=%s" % dict(coverage),
        "confidence_counts=%s" % dict(conf),
        "sample=`%s`" % (metadata[:3],),
    ]
    write_markdown("reports/stage_7_synthesis_summary.md", "Stage 7 Synthesis Summary", [("Results", lines)])
    return {"metadata": len(metadata), "interactions": len(pseudo_interactions), "paths": [meta_path, inter_path], "coverage": dict(coverage), "confidence": dict(conf), "sample": metadata[:3]}
