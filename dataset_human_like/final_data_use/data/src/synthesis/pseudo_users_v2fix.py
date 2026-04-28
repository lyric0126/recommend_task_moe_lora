import os
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table
from src.matching.matcher_v2fix import confidence


def _top_medium(path, min_score):
    rows = read_table(path)
    out = defaultdict(list)
    for r in rows:
        if float(r["score"]) >= min_score:
            out[r["left_user_id"]].append(r)
    for key in out:
        out[key].sort(key=lambda r: r["score"], reverse=True)
    return out


def _global(selected):
    if not selected:
        return 0.0
    scores = [float(r["score"]) for r in selected]
    return round(sum(scores) / len(scores), 6)


def synthesize_v2fix(config):
    syn = config["pipeline"].get("synthesis", {})
    min_component = float(syn.get("min_component_score", 0.5))
    min_global = float(syn.get("min_global_score", 0.5))
    max_users = int(syn.get("max_pseudo_users", 2200))
    max_inter = int(syn.get("max_interactions_per_source_user", 80))
    processed = config["paths"]["processed"]
    maps = {
        "goodreads": _top_medium(os.path.join(processed, "matches_v2fix_movielens_goodreads.parquet"), min_component),
        "mind": _top_medium(os.path.join(processed, "matches_v2fix_movielens_mind.parquet"), min_component),
        "kuairec": _top_medium(os.path.join(processed, "matches_v2fix_movielens_kuairec.parquet"), min_component),
    }
    anchor_ids = sorted(set().union(*[set(m.keys()) for m in maps.values()]))
    metadata = []
    for uid in anchor_ids:
        selected = []
        for domain in ["goodreads", "mind", "kuairec"]:
            if maps[domain].get(uid):
                selected.append((domain, maps[domain][uid][0]))
        selected.sort(key=lambda x: float(x[1]["score"]), reverse=True)
        if len(selected) < 2:
            continue
        for keep in [3, 2]:
            if len(selected) < keep:
                continue
            chosen = selected[:keep]
            gscore = _global([m for _, m in chosen])
            if gscore < min_global:
                continue
            members = {"movielens": uid}
            for domain, match in chosen:
                members[domain] = match["right_user_id"]
            metadata.append({
                "pseudo_user_id": "pseudo_v2fix_%06d" % (len(metadata) + 1),
                "source_members": members,
                "domains_present": sorted(members.keys()),
                "global_consistency_score": gscore,
                "confidence_level": confidence(gscore, config),
                "component_scores": {domain: match["score"] for domain, match in chosen},
            })
            break
        if max_users and len(metadata) >= max_users:
            break
    meta_by_source = defaultdict(list)
    for meta in metadata:
        for ds, uid in meta["source_members"].items():
            meta_by_source[(ds, uid)].append(meta["pseudo_user_id"])
    pseudo_interactions = []
    for ds in ["movielens", "goodreads", "mind", "kuairec"]:
        per_user_counts = Counter()
        for row in read_table(os.path.join(config["paths"]["interim_clean"], ds, "interactions.parquet")):
            key = (ds, row["user_id"])
            pids = meta_by_source.get(key)
            if not pids:
                continue
            if per_user_counts[key] >= max_inter:
                continue
            per_user_counts[key] += 1
            for pid in pids:
                out = dict(row)
                out["pseudo_user_id"] = pid
                out["source_user_id"] = row["user_id"]
                pseudo_interactions.append(out)
    meta_path = os.path.join(processed, "pseudo_user_metadata_v2fix.parquet")
    inter_path = os.path.join(processed, "pseudo_user_interactions_v2fix.parquet")
    write_table(meta_path, metadata, ["pseudo_user_id", "source_members", "domains_present", "global_consistency_score", "confidence_level"], config["pipeline"].get("storage_format", "auto"))
    write_table(inter_path, pseudo_interactions, ["pseudo_user_id", "dataset", "source_user_id", "item_id"], config["pipeline"].get("storage_format", "auto"))
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    conf = Counter(m["confidence_level"] for m in metadata)
    high = sorted(metadata, key=lambda m: m["global_consistency_score"], reverse=True)[:3]
    low = sorted(metadata, key=lambda m: m["global_consistency_score"])[:3]
    v2_count = len(read_table(os.path.join(processed, "pseudo_user_interactions_v2.parquet")))
    lines = [
        "pseudo_users=%s" % len(metadata),
        "pseudo_interactions=%s v2_interactions=%s" % (len(pseudo_interactions), v2_count),
        "coverage=%s" % dict(coverage),
        "confidence=%s" % dict(conf),
        "high_samples=`%s`" % high,
        "low_samples=`%s`" % low,
    ]
    write_markdown("reports/v2fix_synthesis.md", "V2fix Synthesis", [("Smoke Test", lines)])
    return {"paths": [meta_path, inter_path], "metadata": len(metadata), "interactions": len(pseudo_interactions), "coverage": dict(coverage), "confidence": dict(conf)}
