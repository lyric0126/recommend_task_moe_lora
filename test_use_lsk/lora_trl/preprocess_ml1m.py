import os
import json
import random
from collections import defaultdict

random.seed(42)

DATA_DIR = "data/ml-1m"
OUT_DIR = "data/processed_ml1m_mcq"

MOVIES_FILE = os.path.join(DATA_DIR, "movies.dat")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.dat")

MIN_RATING = 4          # 评分>=4视为正反馈
MIN_SEQ_LEN = 5         # 用户至少需要这么多正反馈电影
MAX_HISTORY = 10        # prompt里最多保留多少条历史
NUM_CANDIDATES = 4      # 4选1
LETTERS = ["A", "B", "C", "D"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_movies():
    movie_map = {}
    with open(MOVIES_FILE, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue
            movie_id, title, genres = parts
            movie_map[int(movie_id)] = {
                "title": title,
                "genres": genres,
            }
    return movie_map


def load_positive_sequences():
    """
    返回:
      user_seq[user_id] = [movie_id1, movie_id2, ...]  (按时间升序)
    """
    user_hist = defaultdict(list)

    with open(RATINGS_FILE, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, ts = parts
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)
            ts = int(ts)

            if rating >= MIN_RATING:
                user_hist[user_id].append((ts, movie_id))

    user_seq = {}
    for user_id, items in user_hist.items():
        items.sort(key=lambda x: x[0])
        seq = [movie_id for _, movie_id in items]

        # 去重但保持顺序，避免历史中同一电影反复出现
        dedup_seq = []
        seen = set()
        for m in seq:
            if m not in seen:
                dedup_seq.append(m)
                seen.add(m)

        if len(dedup_seq) >= MIN_SEQ_LEN:
            user_seq[user_id] = dedup_seq

    return user_seq


def build_prompt(history_lines, candidate_lines):
    instruction = (
        "Given the user's movie preference history, choose the most likely next movie "
        "from the candidate list. Return only one letter: A, B, C, or D."
    )
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n"
        f"User history:\n{history_lines}\n\n"
        f"Candidate movies:\n{candidate_lines}\n\n"
        f"### Response:\n"
    )
    return prompt


def sample_negatives(all_movie_ids, forbidden_set, k):
    negatives = []
    while len(negatives) < k:
        m = random.choice(all_movie_ids)
        if m not in forbidden_set and m not in negatives:
            negatives.append(m)
    return negatives


def build_user_samples(user_id, seq, movie_map, all_movie_ids):
    """
    对某个用户，构造多个 prefix->next item 样本
    """
    samples = []

    # 从第4个位置开始构造，保证至少有3条历史
    for i in range(3, len(seq)):
        history = seq[max(0, i - MAX_HISTORY): i]
        target = seq[i]

        if target not in movie_map:
            continue
        if any(h not in movie_map for h in history):
            continue

        forbidden = set(history)
        forbidden.add(target)

        negatives = sample_negatives(all_movie_ids, forbidden, NUM_CANDIDATES - 1)
        candidates = negatives + [target]
        random.shuffle(candidates)

        answer_idx = candidates.index(target)
        answer_letter = LETTERS[answer_idx]

        history_lines = "\n".join(
            f"- {movie_map[h]['title']} | Genres: {movie_map[h]['genres']}"
            for h in history
        )
        candidate_lines = "\n".join(
            f"{LETTERS[j]}. {movie_map[c]['title']} | Genres: {movie_map[c]['genres']}"
            for j, c in enumerate(candidates)
        )

        prompt = build_prompt(history_lines, candidate_lines)

        sample = {
            "user_id": user_id,
            "history_movie_ids": history,
            "target_movie_id": target,
            "candidate_movie_ids": candidates,
            "answer": answer_letter,
            "prompt": prompt,
            "completion": answer_letter,
            "text": prompt + answer_letter,
        }
        samples.append(sample)

    return samples


def split_train_valid_test(all_user_samples):
    train_data, valid_data, test_data = [], [], []

    for user_id, samples in all_user_samples.items():
        # 至少留出 train/valid/test
        if len(samples) < 3:
            continue

        train_data.extend(samples[:-2])
        valid_data.append(samples[-2])
        test_data.append(samples[-1])

    return train_data, valid_data, test_data


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    ensure_dir(OUT_DIR)

    movie_map = load_movies()
    user_seq = load_positive_sequences()
    all_movie_ids = list(movie_map.keys())

    all_user_samples = {}
    total_users = 0

    for user_id, seq in user_seq.items():
        samples = build_user_samples(user_id, seq, movie_map, all_movie_ids)
        if len(samples) >= 3:
            all_user_samples[user_id] = samples
            total_users += 1

    train_data, valid_data, test_data = split_train_valid_test(all_user_samples)

    save_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train_data)
    save_jsonl(os.path.join(OUT_DIR, "valid.jsonl"), valid_data)
    save_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test_data)

    print(f"users kept : {total_users}")
    print(f"train size : {len(train_data)}")
    print(f"valid size : {len(valid_data)}")
    print(f"test size  : {len(test_data)}")
    print(f"saved to   : {OUT_DIR}")

    if train_data:
        print("\n===== Example sample =====")
        print(train_data[0]["prompt"])
        print(train_data[0]["answer"])


if __name__ == "__main__":
    main()