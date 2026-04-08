import os
import json
import random
from collections import defaultdict

random.seed(42)

DATA_DIR = "/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/ml-1m"
OUT_DIR = "/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m"

MOVIES_FILE = os.path.join(DATA_DIR, "movies.dat")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.dat")

MIN_RATING = 4
MIN_SEQ_LEN = 5
MAX_HISTORY = 10
NUM_CANDIDATES = 4
LETTERS = ["A", "B", "C", "D"]

# 给 MovieLens 做一个简单的“伪任务类型”
# 这里按目标电影的主 genre 分桶，作为 task_type
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_movies():
    movie_map = {}
    with open(MOVIES_FILE, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movie_map[int(movie_id)] = {
                "title": title,
                "genres": genres,
            }
    return movie_map


def load_positive_sequences():
    user_hist = defaultdict(list)

    with open(RATINGS_FILE, "r", encoding="latin-1") as f:
        for line in f:
            user_id, movie_id, rating, ts = line.strip().split("::")
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

        dedup_seq = []
        seen = set()
        for m in seq:
            if m not in seen:
                dedup_seq.append(m)
                seen.add(m)

        if len(dedup_seq) >= MIN_SEQ_LEN:
            user_seq[user_id] = dedup_seq

    return user_seq


def sample_negatives(all_movie_ids, forbidden_set, k):
    negatives = []
    while len(negatives) < k:
        m = random.choice(all_movie_ids)
        if m not in forbidden_set and m not in negatives:
            negatives.append(m)
    return negatives


def get_task_type_from_target(movie_info):
    genres = movie_info["genres"].split("|")
    for g in genres:
        if g in GENRE2ID:
            return GENRE2ID[g]
    return 0


def build_instruction_and_input(history, candidates, target_idx, movie_map):
    instruction = (
        "Given the user's movie preference history, choose the most likely next movie "
        "from the candidate list. Answer with only one letter: A, B, C, or D."
    )

    history_lines = "\n".join(
        f"- {movie_map[h]['title']} | Genres: {movie_map[h]['genres']}"
        for h in history
    )
    candidate_lines = "\n".join(
        f"{LETTERS[j]}. {movie_map[c]['title']} | Genres: {movie_map[c]['genres']}"
        for j, c in enumerate(candidates)
    )

    input_text = (
        f"User history:\n{history_lines}\n\n"
        f"Candidate movies:\n{candidate_lines}"
    )

    output = LETTERS[target_idx]
    return instruction, input_text, output


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ensure_dir(OUT_DIR)

    movie_map = load_movies()
    user_seq = load_positive_sequences()
    all_movie_ids = list(movie_map.keys())

    train_data = []
    valid_data = []

    for user_id, seq in user_seq.items():
        user_samples = []

        for i in range(3, len(seq)):
            history = seq[max(0, i - MAX_HISTORY): i]
            target = seq[i]

            if target not in movie_map:
                continue

            forbidden = set(history)
            forbidden.add(target)

            negatives = sample_negatives(all_movie_ids, forbidden, NUM_CANDIDATES - 1)
            candidates = negatives + [target]
            random.shuffle(candidates)

            target_idx = candidates.index(target)
            instruction, input_text, output = build_instruction_and_input(
                history, candidates, target_idx, movie_map
            )

            task_type = get_task_type_from_target(movie_map[target])

            sample = {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "task_type": task_type,
                "user_id": user_id,
                "target_movie_id": target,
                "candidate_movie_ids": candidates,
            }
            user_samples.append(sample)

        if len(user_samples) >= 3:
            train_data.extend(user_samples[:-2])
            valid_data.append(user_samples[-2])

    save_json(os.path.join(OUT_DIR, "train.json"), train_data)
    save_json(os.path.join(OUT_DIR, "valid.json"), valid_data)

    print(f"train size: {len(train_data)}")
    print(f"valid size: {len(valid_data)}")
    print(f"saved to  : {OUT_DIR}")

    if train_data:
        print("\nExample:")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()