import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# 如果后面换 llama，本地路径改这里：
# MODEL_NAME = "/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B-Instruct"

DATA_DIR = "data/processed_ml1m_mcq"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VALID_PATH = os.path.join(DATA_DIR, "valid.jsonl")

OUTPUT_DIR = "outputs/trl_lora_ml1m_qwen"


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = load_jsonl(TRAIN_PATH)
    valid_data = load_jsonl(VALID_PATH)

    print(f"Loaded train samples: {len(train_data)}")
    print(f"Loaded valid samples: {len(valid_data)}")

    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=os.path.join(OUTPUT_DIR, "tb_logs"),
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    save_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Saved adapter to: {save_dir}")


if __name__ == "__main__":
    main()