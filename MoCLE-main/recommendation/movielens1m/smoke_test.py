#!/usr/bin/env python3
"""Run a minimal MovieLens-1M recommendation training/forward smoke test."""

import argparse
import os
import sys


if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from recommendation.movielens1m.dataset import (  # noqa: E402
        PurePythonUnigramLM,
        SimpleByteTokenizer,
        build_features,
        collate_features,
        load_instruction_samples,
    )
else:
    from .dataset import (  # noqa: E402
        PurePythonUnigramLM,
        SimpleByteTokenizer,
        build_features,
        collate_features,
        load_instruction_samples,
    )


DEFAULT_DATA_PATH = "data/movielens1m_smoke/train.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=["auto", "python", "torch"],
        default="auto",
        help="auto uses torch+transformers when available, otherwise pure Python.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B",
        help="Tokenizer path for the torch backend.",
    )
    return parser.parse_args()


def have_torch_backend():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def run_python_backend(samples, args):
    tokenizer = SimpleByteTokenizer()
    features = build_features(samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    batch = collate_features(features[: args.batch_size], pad_token_id=tokenizer.pad_token_id)
    supervised_tokens = sum(x != -100 for row in batch["labels"] for x in row)
    if supervised_tokens == 0:
        raise ValueError(
            "No supervised tokens survived truncation. Increase --max_seq_length "
            "or reduce prompt length."
        )

    model = PurePythonUnigramLM(vocab_size=tokenizer.vocab_size)
    initial_loss = model.loss(batch)
    for _ in range(args.train_steps):
        model.train_batch(batch)
    final_loss = model.loss(batch)

    print("backend              : pure-python unigram LM")
    print("batch input shape    : {} x {}".format(len(batch["input_ids"]), len(batch["input_ids"][0])))
    print("supervised tokens    : {}".format(supervised_tokens))
    print("initial loss         : {:.6f}".format(initial_loss))
    print("final loss           : {:.6f}".format(final_loss))

    token_id = model.most_likely_token()
    decoded = tokenizer.decode([token_id]) if token_id is not None else ""
    print("most likely token    : {!r}".format(decoded))
    print("first sample output  : {}".format(samples[0]["output"]))


def run_torch_backend(samples, args):
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    features = build_features(samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    batch = collate_features(features[: args.batch_size], pad_token_id=tokenizer.pad_token_id, return_tensors=True)
    supervised_tokens = int(batch["labels"].ne(-100).sum().item())
    if supervised_tokens == 0:
        raise ValueError(
            "No supervised tokens survived truncation. Increase --max_seq_length "
            "or reduce prompt length."
        )

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=max(args.max_seq_length, 64),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }

    with torch.no_grad():
        initial_loss = model(**model_inputs).loss.item()
    final_loss = initial_loss
    for _ in range(args.train_steps):
        optimizer.zero_grad()
        loss = model(**model_inputs).loss
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    print("backend              : torch tiny LlamaForCausalLM")
    print("tokenizer            : {}".format(args.tokenizer_name_or_path))
    print("batch input shape    : {}".format(tuple(batch["input_ids"].shape)))
    print("supervised tokens    : {}".format(supervised_tokens))
    print("initial loss         : {:.6f}".format(initial_loss))
    print("final loss           : {:.6f}".format(final_loss))
    print("first sample output  : {}".format(samples[0]["output"]))


def main():
    args = parse_args()
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError("Smoke data file not found: {}".format(args.data_path))

    samples = load_instruction_samples(args.data_path)[: args.max_samples]
    if not samples:
        raise ValueError("No samples loaded from {}".format(args.data_path))

    print("data_path            : {}".format(os.path.abspath(args.data_path)))
    print("loaded samples       : {}".format(len(samples)))
    print("first sample user_id : {}".format(samples[0].get("user_id")))
    print("first target movie   : {}".format(samples[0].get("target_movie_id")))

    use_torch = args.backend == "torch" or (args.backend == "auto" and have_torch_backend())
    if use_torch:
        run_torch_backend(samples, args)
    else:
        run_python_backend(samples, args)

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
