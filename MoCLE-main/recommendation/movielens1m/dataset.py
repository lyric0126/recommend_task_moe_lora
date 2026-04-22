"""Dataset helpers for HydraLoRA-style instruction recommendation samples."""

import json
import math
from collections import Counter


IGNORE_INDEX = -100
PROMPT_TEMPLATE = "{instruction}</s>"


def load_instruction_samples(path):
    if path.endswith(".jsonl"):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    for idx, sample in enumerate(samples):
        missing = {"instruction", "output"} - set(sample)
        if missing:
            raise ValueError("Sample {} in {} missing keys: {}".format(idx, path, sorted(missing)))
        sample.setdefault("input", "")
        sample.setdefault("cluster_id", sample.get("task_type", 0))
    return samples


def make_source_target(sample, eos_token="</s>"):
    instruction = sample["instruction"]
    input_text = sample.get("input") or ""
    if input_text:
        instruction = instruction + "\n" + input_text
    source = PROMPT_TEMPLATE.format_map({"instruction": instruction})
    target = "{}{}".format(sample["output"], eos_token)
    return source, target


class SimpleByteTokenizer:
    """Small deterministic tokenizer used when Transformers is unavailable."""

    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"
    vocab_size = 258

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_eos=False):
        ids = [byte + 2 for byte in text.encode("utf-8")]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        byte_values = []
        for token_id in ids:
            if token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            if 2 <= token_id < self.vocab_size:
                byte_values.append(token_id - 2)
        return bytes(byte_values).decode("utf-8", errors="replace")


def encode_with_tokenizer(tokenizer, text, add_special_tokens=False):
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            return tokenizer.encode(text)
    encoded = tokenizer(text, return_attention_mask=False, add_special_tokens=add_special_tokens)
    return encoded["input_ids"]


def truncate_source_target(source_ids, target_ids, max_seq_length):
    """Keep target labels when truncating long recommendation prompts."""
    if max_seq_length is None or max_seq_length <= 0:
        return list(source_ids) + list(target_ids), [-100] * len(source_ids) + list(target_ids)

    target_ids = list(target_ids)
    if len(target_ids) >= max_seq_length:
        input_ids = target_ids[:max_seq_length]
        labels = target_ids[:max_seq_length]
        return input_ids, labels

    max_source_len = max_seq_length - len(target_ids)
    source_ids = list(source_ids)[-max_source_len:] if max_source_len > 0 else []
    input_ids = source_ids + target_ids
    labels = [IGNORE_INDEX] * len(source_ids) + target_ids
    return input_ids, labels


def build_features(samples, tokenizer=None, max_seq_length=512):
    tokenizer = tokenizer or SimpleByteTokenizer()
    eos_token = getattr(tokenizer, "eos_token", None) or "</s>"
    features = []

    for sample in samples:
        source, target = make_source_target(sample, eos_token=eos_token)
        source_ids = encode_with_tokenizer(tokenizer, source, add_special_tokens=False)
        target_ids = encode_with_tokenizer(tokenizer, target, add_special_tokens=False)
        input_ids, labels = truncate_source_target(source_ids, target_ids, max_seq_length)
        features.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
                "meta": sample,
            }
        )
    return features


def collate_features(features, pad_token_id=0, return_tensors=False):
    max_len = max(len(item["input_ids"]) for item in features)
    input_ids, labels, attention_mask, cluster_id = [], [], [], []
    for item in features:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        cluster_id.append(int(item.get("cluster_id", 0)))

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "cluster_id": cluster_id,
    }
    if return_tensors:
        import torch

        batch = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return batch


class InstructionDataset:
    """Lazy tokenized dataset for PyTorch training."""

    def __init__(self, samples, tokenizer, max_seq_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.eos_token = getattr(tokenizer, "eos_token", None) or "</s>"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source, target = make_source_target(sample, eos_token=self.eos_token)
        source_ids = encode_with_tokenizer(self.tokenizer, source, add_special_tokens=False)
        target_ids = encode_with_tokenizer(self.tokenizer, target, add_special_tokens=False)
        input_ids, labels = truncate_source_target(source_ids, target_ids, self.max_seq_length)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "cluster_id": sample["cluster_id"],
            "meta": sample,
        }


def make_torch_collate_fn(pad_token_id):
    def collate(instances):
        return collate_features(instances, pad_token_id=pad_token_id, return_tensors=True)

    return collate


class PurePythonUnigramLM:
    """A tiny trainable language-model smoke backend with no external deps."""

    def __init__(self, vocab_size=258, smoothing=1.0):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = Counter()
        self.total = 0

    def train_batch(self, batch):
        for row in batch["labels"]:
            for token_id in row:
                if token_id != IGNORE_INDEX:
                    self.counts[token_id] += 1
                    self.total += 1

    def loss(self, batch):
        denom = self.total + self.smoothing * self.vocab_size
        if denom <= 0:
            return float("inf")

        nll = 0.0
        count = 0
        for row in batch["labels"]:
            for token_id in row:
                if token_id == IGNORE_INDEX:
                    continue
                prob = (self.counts[token_id] + self.smoothing) / denom
                nll -= math.log(prob)
                count += 1
        return nll / max(count, 1)

    def most_likely_token(self):
        if not self.counts:
            return None
        return self.counts.most_common(1)[0][0]
