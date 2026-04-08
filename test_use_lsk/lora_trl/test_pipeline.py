from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train[:100]")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=SFTConfig(
        output_dir="./trl_test_out",
        per_device_train_batch_size=1,
        max_steps=5,
        logging_steps=1,
        report_to="none",
    ),
    train_dataset=dataset,
)

trainer.train()
print("TRL SFTTrainer test passed.")