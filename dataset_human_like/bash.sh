#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD/data/raw}"
mkdir -p "$ROOT"/{movielens,goodreads,mind,kuairec}

download() {
  local url="$1"
  local out_dir="$2"

  cd "$out_dir"

  if command -v aria2c >/dev/null 2>&1; then
    echo "[aria2c] $url"
    aria2c -x 16 -s 16 -k 1M -c "$url"
  elif command -v wget >/dev/null 2>&1; then
    echo "[wget] $url"
    wget -c --tries=20 --timeout=30 --read-timeout=30 "$url"
  else
    echo "Neither aria2c nor wget is installed."
    exit 1
  fi
}

echo "Downloading into: $ROOT"

# 1) MovieLens
download "https://files.grouplens.org/datasets/movielens/ml-25m.zip" "$ROOT/movielens"

# 2) Goodreads
download "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_young_adult.json.gz" "$ROOT/goodreads"
download "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_young_adult.json.gz" "$ROOT/goodreads"
download "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz" "$ROOT/goodreads"

# 3) MIND
download "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip" "$ROOT/mind"
download "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip" "$ROOT/mind"

# 4) KuaiRec
download "https://zenodo.org/records/18164998/files/KuaiRec.zip" "$ROOT/kuairec"
download "https://zenodo.org/records/18164998/files/kuairec_caption_category.csv" "$ROOT/kuairec"

echo "All downloads finished."
