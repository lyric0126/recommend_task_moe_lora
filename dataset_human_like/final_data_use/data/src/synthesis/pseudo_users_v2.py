import os
from collections import Counter, defaultdict

from src.io_utils import read_table, write_markdown, write_table
from src.matching.matcher_v2 import confidence


def _top_map(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["left_user_id"]].append(r)
    for key in out:
        out[key].sort(key=lambda r: r["score"], reverse=True)
    return out


def _global_score(selected):
    if not selected:
        return 0.0
    scores = [float(r["score"]) for r in selected]
    mean = sum(scores) / len(scores)
    penalty = 0.03 * max(0, 3 - len(selected))
    return round(max(0.0, mean - penalty), 6)


def synthesize_v2(config):
    ml_gr = _top_map(read_table(os.path.join(config["paths"]["processed"], "matches_v2_movielens_goodreads.parquet")))
    ml_mind = _top_map(read_table(os.path.join(config["paths"]["processed"], "matches_v2_movielens_mind.parquet")))
    ml_kr = _top_map(read_table(os.path.join(config["paths"]["processed"], "matches_v2_movielens_kuairec.parquet")))
    anchor_ids = sorted(set(ml_gr.keys()) | set(ml_mind.keys()) | set(ml_kr.keys()))
    max_anchor = int(config["pipeline"].get("synthesis", {}).get("max_anchor_users", 0) or 0)
    if max_anchor:
        anchor_ids = anchor_ids[:max_anchor]
    min_domains = int(config["pipeline"].get("synthesis", {}).get("min_domains", 2))
    metadata = []
    for idx, uid in enumerate(anchor_ids, 1):
        candidates = []
        for domain, mapping in [("goodreads", ml_gr), ("mind", ml_mind), ("kuairec", ml_kr)]:
            if mapping.get(uid):
                candidates.append((domain, mapping[uid][0]))
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        if len(candidates) + 1 < min_domains:
            continue
        for keep in range(min_domains - 1, min(3, len(candidates)) + 1):
            selected = candidates[:keep]
            source_members = {"movielens": uid}
            for domain, match in selected:
                source_members[domain] = match["right_user_id"]
            score = _global_score([m for _, m in selected])
            metadata.append({
                "pseudo_user_id": "pseudo_v2_%06d_%dd" % (idx, len(source_members)),
                "source_members": source_members,
                "domains_present": sorted(source_members.keys()),
                "global_consistency_score": score,
                "confidence_level": confidence(score, config),
                "component_scores": {domain: match["score"] for domain, match in selected},
            })
    source_lookup = {}
    for meta in metadata:
        for ds, uid in meta["source_members"].items():
            source_lookup[(ds, uid, meta["pseudo_user_id"])] = True
    pseudo_interactions = []
    meta_by_source = defaultdict(list)
    for meta in metadata:
        for ds, uid in meta["source_members"].items():
            meta_by_source[(ds, uid)].append(meta["pseudo_user_id"])
    for dataset in ["movielens", "goodreads", "mind", "kuairec"]:
        for row in read_table(os.path.join(config["paths"]["interim_clean"], dataset, "interactions.parquet")):
            for pid in meta_by_source.get((dataset, row["user_id"]), []):
                new_row = dict(row)
                new_row["pseudo_user_id"] = pid
                new_row["source_user_id"] = row["user_id"]
                pseudo_interactions.append(new_row)
    meta_path = os.path.join(config["paths"]["processed"], "pseudo_user_metadata_v2.parquet")
    inter_path = os.path.join(config["paths"]["processed"], "pseudo_user_interactions_v2.parquet")
    write_table(meta_path, metadata, ["pseudo_user_id", "source_members", "domains_present", "global_consistency_score", "confidence_level"], config["pipeline"].get("storage_format", "auto"))
    write_table(inter_path, pseudo_interactions, ["pseudo_user_id", "dataset", "source_user_id", "item_id"], config["pipeline"].get("storage_format", "auto"))
    coverage = Counter(len(m["domains_present"]) for m in metadata)
    conf = Counter(m["confidence_level"] for m in metadata)
    high = sorted(metadata, key=lambda m: m["global_consistency_score"], reverse=True)[:3]
    low = sorted(metadata, key=lambda m: m["global_consistency_score"])[:3]
    lines = [
        "pseudo_users=%s" % len(metadata),
        "pseudo_interactions=%s" % len(pseudo_interactions),
        "coverage=%s" % dict(coverage),
        "confidence=%s" % dict(conf),
        "high_confidence_samples=`%s`" % high,
        "low_confidence_samples=`%s`" % low,
    ]
    write_markdown("reports/v2_synthesis_upgrade.md", "V2 Synthesis Upgrade", [("Smoke Test", lines)])
    return {"paths": [meta_path, inter_path], "metadata": len(metadata), "interactions": len(pseudo_interactions), "coverage": dict(coverage), "confidence": dict(conf)}
