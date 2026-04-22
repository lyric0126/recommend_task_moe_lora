#!/usr/bin/env python3
"""Build MovieLens-1M next-movie recommendation samples.

The output JSON format is intentionally compatible with the HydraLoRA training
path: every sample has at least ``instruction``, ``input`` and ``output``.
"""

import argparse
import json
import os
import random
from collections import defaultdict


DEFAULT_RAW_DIR = "/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/ml-1m"
DEFAULT_OUT_DIR = "data/movielens1m"
LETTERS = ["A", "B", "C", "D"]

GENRE2ID = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children's": 3,
    "Comedy": 4,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "Musical": 11,
    "Mystery": 12,
    "Romance": 13,
    "Sci-Fi": 14,
    "Thriller": 15,
    "War": 16,
    "Western": 17,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR, help="Directory containing ml-1m *.dat files.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Directory to write processed files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_rating", type=int, default=4)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument(
        "--format",
        choices=["hydralora", "jsonl", "both"],
        default="both",
        help="hydralora writes train/valid/test.json; jsonl writes train/valid/test.jsonl.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Required MovieLens-1M file is missing: {}\n"
            "Expected raw_dir to contain movies.dat, ratings.dat and users.dat.".format(path)
        )


def load_movies(raw_dir):
    movies_file = os.path.join(raw_dir, "movies.dat")
    require_file(movies_file)

    movie_map = {}
    with open(movies_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            movie_id, title, genres = parts
            movie_map[int(movie_id)] = {"title": title, "genres": genres}
    return movie_map


def load_positive_sequences(raw_dir, min_rating, min_seq_len):
    ratings_file = os.path.join(raw_dir, "ratings.dat")
    require_file(ratings_file)

    user_hist = defaultdict(list)
    with open(ratings_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, ts = parts
            if int(rating) >= min_rating:
                user_hist[int(user_id)].append((int(ts), int(movie_id)))

    user_seq = {}
    for user_id, items in user_hist.items():
        items.sort(key=lambda x: x[0])
        dedup_seq = []
        seen = set()
        for _, movie_id in items:
            if movie_id not in seen:
                dedup_seq.append(movie_id)
                seen.add(movie_id)
        if len(dedup_seq) >= min_seq_len:
            user_seq[user_id] = dedup_seq
    return user_seq


def sample_negatives(rng, all_movie_ids, forbidden_set, k):
    negatives = []
    while len(negatives) < k:
        movie_id = rng.choice(all_movie_ids)
        if movie_id not in forbidden_set and movie_id not in negatives:
            negatives.append(movie_id)
    return negatives


def get_task_type(movie_info):
    for genre in movie_info["genres"].split("|"):
        if genre in GENRE2ID:
            return GENRE2ID[genre]
    return 0


def build_prompt_parts(history, candidates, answer_idx, movie_map):
    instruction = (
        "Given the user's movie preference history, choose the most likely next movie "
        "from the candidate list. Answer with only one letter: A, B, C, or D."
    )
    history_lines = "\n".join(
        "- {} | Genres: {}".format(movie_map[movie_id]["title"], movie_map[movie_id]["genres"])
        for movie_id in history
    )
    candidate_lines = "\n".join(
        "{}. {} | Genres: {}".format(
            LETTERS[idx], movie_map[movie_id]["title"], movie_map[movie_id]["genres"]
        )
        for idx, movie_id in enumerate(candidates)
    )
    input_text = "User history:\n{}\n\nCandidate movies:\n{}".format(history_lines, candidate_lines)
    output = LETTERS[answer_idx]
    prompt = (
        "### Instruction:\n{}\n\n"
        "### Input:\n{}\n\n"
        "### Response:\n"
    ).format(instruction, input_text)
    return instruction, input_text, output, prompt


def build_user_samples(user_id, seq, movie_map, all_movie_ids, rng, args):
    samples = []
    for idx in range(3, len(seq)):
        history = seq[max(0, idx - args.max_history) : idx]
        target = seq[idx]

        if target not in movie_map or any(movie_id not in movie_map for movie_id in history):
            continue

        forbidden = set(history)
        forbidden.add(target)
        negatives = sample_negatives(rng, all_movie_ids, forbidden, args.num_candidates - 1)
        candidates = negatives + [target]
        rng.shuffle(candidates)

        answer_idx = candidates.index(target)
        instruction, input_text, output, prompt = build_prompt_parts(history, candidates, answer_idx, movie_map)
        task_type = get_task_type(movie_map[target])
        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "task_type": task_type,
            "cluster_id": task_type,
            "user_id": user_id,
            "history_movie_ids": history,
            "target_movie_id": target,
            "candidate_movie_ids": candidates,
            "prompt": prompt,
            "completion": output,
            "text": prompt + output,
        }
        samples.append(sample)
    return samples


def split_by_user(user_samples):
    train_data, valid_data, test_data = [], [], []
    for samples in user_samples.values():
        if len(samples) < 3:
            continue
        train_data.extend(samples[:-2])
        valid_data.append(samples[-2])
        test_data.append(samples[-1])
    return train_data, valid_data, test_data


def hydralora_view(sample):
    return {
        "instruction": sample["instruction"],
        "input": sample["input"],
        "output": sample["output"],
        "task_type": sample["task_type"],
        "cluster_id": sample["cluster_id"],
        "user_id": sample["user_id"],
        "history_movie_ids": sample["history_movie_ids"],
        "target_movie_id": sample["target_movie_id"],
        "candidate_movie_ids": sample["candidate_movie_ids"],
    }


def limit(data, max_items):
    if max_items is None:
        return data
    return data[:max_items]


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.num_candidates > len(LETTERS):
        raise ValueError("--num_candidates cannot exceed {} for A-D labels".format(len(LETTERS)))

    rng = random.Random(args.seed)
    ensure_dir(args.out_dir)

    movie_map = load_movies(args.raw_dir)
    user_seq = load_positive_sequences(args.raw_dir, args.min_rating, args.min_seq_len)
    all_movie_ids = list(movie_map.keys())

    all_user_samples = {}
    for user_id in sorted(user_seq):
        samples = build_user_samples(user_id, user_seq[user_id], movie_map, all_movie_ids, rng, args)
        if len(samples) >= 3:
            all_user_samples[user_id] = samples

    train_data, valid_data, test_data = split_by_user(all_user_samples)
    train_data = limit(train_data, args.max_train_samples)
    valid_data = limit(valid_data, args.max_valid_samples)
    test_data = limit(test_data, args.max_test_samples)

    if args.format in {"hydralora", "both"}:
        save_json(os.path.join(args.out_dir, "train.json"), [hydralora_view(x) for x in train_data])
        save_json(os.path.join(args.out_dir, "valid.json"), [hydralora_view(x) for x in valid_data])
        save_json(os.path.join(args.out_dir, "test.json"), [hydralora_view(x) for x in test_data])

    if args.format in {"jsonl", "both"}:
        save_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_data)
        save_jsonl(os.path.join(args.out_dir, "valid.jsonl"), valid_data)
        save_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_data)

    print("raw_dir    : {}".format(os.path.abspath(args.raw_dir)))
    print("out_dir    : {}".format(os.path.abspath(args.out_dir)))
    print("users kept : {}".format(len(all_user_samples)))
    print("train size : {}".format(len(train_data)))
    print("valid size : {}".format(len(valid_data)))
    print("test size  : {}".format(len(test_data)))
    if train_data:
        print("\n===== Example sample =====")
        print(json.dumps(hydralora_view(train_data[0]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
