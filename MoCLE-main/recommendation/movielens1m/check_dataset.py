#!/usr/bin/env python3
"""Check MovieLens-1M recommendation samples and one collated batch."""

import argparse
import os
import sys


if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recommendation.movielens1m.dataset import (  # noqa: E402
    InstructionDataset,
    SimpleByteTokenizer,
    collate_features,
    load_instruction_samples,
)


DEFAULT_DATA_PATH = "data/movielens1m_train_debug/train.json"
DEFAULT_TOKENIZER = "/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--backend", choices=["auto", "simple", "transformers"], default="auto")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()


def load_tokenizer(args):
    if args.backend in {"auto", "transformers"}:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, "transformers"
        except Exception:
            if args.backend == "transformers":
                raise
    return SimpleByteTokenizer(), "simple-byte"


def shape_of(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, list) and value:
        return (len(value), len(value[0]))
    if isinstance(value, list):
        return (0,)
    return type(value).__name__


def main():
    args = parse_args()
    samples = load_instruction_samples(args.data_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples found in {}".format(args.data_path))

    tokenizer, backend = load_tokenizer(args)
    dataset = InstructionDataset(samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    instances = [dataset[idx] for idx in range(min(args.batch_size, len(dataset)))]
    batch = collate_features(instances, pad_token_id=tokenizer.pad_token_id, return_tensors=False)
    supervised_tokens = sum(token_id != -100 for row in batch["labels"] for token_id in row)
    if supervised_tokens == 0:
        raise ValueError("No supervised tokens in checked batch; increase --max_seq_length.")

    print("data_path         : {}".format(os.path.abspath(args.data_path)))
    print("tokenizer backend : {}".format(backend))
    print("samples loaded    : {}".format(len(samples)))
    print("sample keys       : {}".format(sorted(samples[0].keys())))
    print("first user_id     : {}".format(samples[0].get("user_id")))
    print("first target      : {}".format(samples[0].get("target_movie_id")))
    print("first cluster_id  : {}".format(samples[0].get("cluster_id")))
    print("batch keys        : {}".format(sorted(batch.keys())))
    print("input_ids shape   : {}".format(shape_of(batch["input_ids"])))
    print("labels shape      : {}".format(shape_of(batch["labels"])))
    print("attention shape   : {}".format(shape_of(batch["attention_mask"])))
    print("batch cluster_id  : {}".format(batch.get("cluster_id")))
    print("supervised tokens : {}".format(supervised_tokens))
    print("DATASET_CHECK_OK")


if __name__ == "__main__":
    main()
