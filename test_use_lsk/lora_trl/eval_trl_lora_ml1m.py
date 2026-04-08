import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# 如果后面换 llama，本地路径和训练脚本保持一致即可

ADAPTER_PATH = "outputs/trl_lora_ml1m_qwen/final_adapter"
TEST_PATH = "data/processed_ml1m_mcq/test.jsonl"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_letter(text: str):
    text = text.strip().upper()
    for ch in text:
        if ch in ["A", "B", "C", "D"]:
            return ch
    return None


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    test_data = load_jsonl(TEST_PATH)

    correct = 0
    total = 0

    for idx, item in enumerate(tqdm(test_data)):
        prompt = item["prompt"]
        gold = item["answer"]

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        pred = extract_letter(gen)

        if pred == gold:
            correct += 1
        total += 1

        if idx < 5:
            print("=" * 80)
            print(prompt)
            print("GEN :", gen)
            print("PRED:", pred, " GOLD:", gold)

    acc = correct / total if total > 0 else 0.0
    print(f"\nTest Accuracy / Hit@1 = {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()