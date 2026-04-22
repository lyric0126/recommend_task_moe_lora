#!/usr/bin/env python3
"""Single-GPU MovieLens-1M recommendation training entrypoint."""

import argparse
import json
import os
import sys
import time


if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recommendation.movielens1m.dataset import (  # noqa: E402
    InstructionDataset,
    load_instruction_samples,
    make_torch_collate_fn,
)


DEFAULT_MODEL = "/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B"
DEFAULT_TRAIN_FILE = "data/movielens1m_train_debug/train.json"
DEFAULT_OUTPUT_DIR = "outputs/movielens1m_mocle_small"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50, help="Save adapter checkpoints every N steps; <=0 disables periodic saves.")
    parser.add_argument("--tensorboard_log_dir", default=None)
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--train_mode",
        choices=["lm_head", "full", "lora", "mocle"],
        default="lm_head",
        help="lm_head is the safest debug mode; lora/mocle use local peft-main when available.",
    )
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--save_model", action="store_true")
    return parser.parse_args()


def require_training_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Single-GPU training requires torch and transformers.") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def dtype_from_arg(torch, value):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def load_training_tokenizer(AutoTokenizer, args):
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer_path


def build_lora_config(args):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


def ensure_local_peft():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    peft_src = os.path.join(repo_root, "peft-main", "src")
    if peft_src not in sys.path:
        sys.path.insert(0, peft_src)


def build_mocle_model(model, args):
    if args.num_experts <= 0:
        raise ValueError("--num_experts must be positive")
    ensure_local_peft()
    try:
        from peft import get_peft_model
    except Exception as exc:
        raise RuntimeError("Unable to import local PEFT for --train_mode mocle: {}".format(exc)) from exc

    peft_config = build_lora_config(args)
    model = get_peft_model(model, peft_config, adapter_name="expert_0")
    for expert_idx in range(1, args.num_experts):
        model.add_adapter("expert_{}".format(expert_idx), peft_config)
    model.set_adapter("expert_0")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def set_train_mode(model, args):
    if args.train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return model, "full"

    if args.train_mode == "lm_head":
        for param in model.parameters():
            param.requires_grad = False
        if not hasattr(model, "lm_head"):
            raise ValueError("Model has no lm_head; use --train_mode full.")
        for param in model.lm_head.parameters():
            param.requires_grad = True
        return model, "lm_head"

    if args.train_mode == "mocle":
        return build_mocle_model(model, args), "mocle"

    ensure_local_peft()
    try:
        from peft import get_peft_model
    except Exception as exc:
        raise RuntimeError("Unable to import local PEFT for --train_mode lora: {}".format(exc)) from exc

    peft_config = build_lora_config(args)
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, "lora"


def count_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def create_summary_writer(args):
    if args.disable_tensorboard:
        return None, None
    log_dir = args.tensorboard_log_dir or os.path.join(args.output_dir, "tb_logs")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but tensorboard is not installed in this Python environment. "
            "Install it in lavispy310, for example: pip install tensorboard"
        ) from exc
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir), log_dir


def compute_grad_norm(torch, parameters):
    norm_sq = None
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = torch.linalg.vector_norm(param.grad.detach().float(), ord=2)
        norm_sq = param_norm.pow(2) if norm_sq is None else norm_sq + param_norm.pow(2)
    if norm_sq is None:
        return 0.0
    return float(torch.sqrt(norm_sq).item())


def write_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_step_checkpoint(model, tokenizer, args, step):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-step-{}".format(step))
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print("checkpoint saved     : {}".format(checkpoint_dir))


def extract_first_cluster_id(batch):
    cluster_ids = batch.get("cluster_id")
    if cluster_ids is None:
        return None
    if hasattr(cluster_ids, "view"):
        return int(cluster_ids.view(-1)[0].item())
    if isinstance(cluster_ids, (list, tuple)):
        return int(cluster_ids[0])
    return int(cluster_ids)


def make_model_inputs(batch, device):
    allowed = {"input_ids", "attention_mask", "labels"}
    return {key: value.to(device) for key, value in batch.items() if key in allowed}


def main():
    args = parse_args()
    torch, AutoModelForCausalLM, AutoTokenizer = require_training_deps()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check the lavispy310 PyTorch/CUDA build against the installed NVIDIA driver."
        )
    device = torch.device(requested_device)

    tokenizer, tokenizer_path = load_training_tokenizer(AutoTokenizer, args)

    samples = load_instruction_samples(args.train_file)
    if args.max_train_samples is not None:
        samples = samples[: args.max_train_samples]
    if not samples:
        raise ValueError("No training samples found in {}".format(args.train_file))

    dataset = InstructionDataset(samples, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    collate_fn = make_torch_collate_fn(tokenizer.pad_token_id)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    first_batch = next(iter(dataloader))
    supervised_tokens = int(first_batch["labels"].ne(-100).sum().item())
    if supervised_tokens == 0:
        raise ValueError("No supervised tokens in first batch; increase --max_seq_length.")

    dtype = dtype_from_arg(torch, args.torch_dtype)
    model_kwargs = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    print("repo cwd             : {}".format(os.getcwd()))
    print("train_file           : {}".format(os.path.abspath(args.train_file)))
    print("output_dir           : {}".format(os.path.abspath(args.output_dir)))
    print("tensorboard_log_dir  : {}".format(os.path.abspath(args.tensorboard_log_dir or os.path.join(args.output_dir, "tb_logs"))))
    print("model                : {}".format(args.model_name_or_path))
    print("tokenizer            : {}".format(tokenizer_path))
    print("requested device     : {}".format(args.device))
    print("actual device        : {}".format(device))
    if torch.cuda.is_available():
        print("cuda visible devices : {}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
        print("gpu name             : {}".format(torch.cuda.get_device_name(0)))
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print("gpu memory free/total: {:.2f} GiB / {:.2f} GiB".format(free_bytes / 2**30, total_bytes / 2**30))
    print("samples loaded       : {}".format(len(samples)))
    print("batch keys           : {}".format(sorted(first_batch.keys())))
    print("batch input_ids shape: {}".format(tuple(first_batch["input_ids"].shape)))
    print("batch labels shape   : {}".format(tuple(first_batch["labels"].shape)))
    first_batch_cluster_id = extract_first_cluster_id(first_batch)
    print("first batch cluster_id: {}".format(first_batch_cluster_id))
    print("num experts          : {}".format(args.num_experts))
    print("max steps            : {}".format(args.max_steps))
    print("logging steps        : {}".format(args.logging_steps))
    print("save steps           : {}".format(args.save_steps))
    print("supervised tokens    : {}".format(supervised_tokens))
    print("loading model...")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    print("model init           : from_pretrained")
    if hasattr(model, "config"):
        model.config.use_cache = False
    model, active_mode = set_train_mode(model, args)
    model.to(device)
    model.train()

    total_params, trainable_params = count_parameters(model)
    print("train mode           : {}".format(active_mode))
    print("total params         : {}".format(total_params))
    print("trainable params     : {}".format(trainable_params))
    if trainable_params == 0:
        raise ValueError("No trainable parameters selected.")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    writer, tensorboard_log_dir = create_summary_writer(args)
    metrics_path = os.path.join(args.output_dir, "train_metrics.jsonl")

    global_step = 0
    last_loss = None
    last_lr = None
    last_grad_norm = None
    last_step_time = None
    last_cluster_id = None
    last_expert_id = None
    last_active_expert = None
    last_seq_len = None
    last_tokens_per_step = None
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < args.max_steps:
            for batch_idx, batch in enumerate(dataloader):
                step_start = time.time()
                active_expert = None
                routed_cluster_id = None
                cluster_id = extract_first_cluster_id(batch)
                if active_mode == "mocle":
                    if args.batch_size != 1:
                        raise ValueError("MovieLens MoCLE-v0 only supports --batch_size 1.")
                    routed_cluster_id = cluster_id % args.num_experts
                    active_expert = "expert_{}".format(routed_cluster_id)
                    model.set_adapter(active_expert)
                    if global_step == 0 or args.logging_steps <= 1:
                        print(
                            "MOCLE_ROUTE step={} raw_cluster_id={} expert={}".format(
                                global_step + 1, cluster_id, active_expert
                            )
                        )
                model_inputs = make_model_inputs(batch, device)
                outputs = model(**model_inputs)
                raw_loss = outputs.loss
                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    grad_norm = compute_grad_norm(torch, trainable_parameters)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    step_time = time.time() - step_start
                    current_lr = optimizer.param_groups[0]["lr"]
                    seq_len = int(model_inputs["input_ids"].shape[1])
                    tokens_per_step = int(model_inputs["attention_mask"].sum().item())
                    expert_id = -1 if routed_cluster_id is None else int(routed_cluster_id)
                    loss_value = float(raw_loss.detach().float().item())

                    last_loss = loss_value
                    last_lr = current_lr
                    last_grad_norm = grad_norm
                    last_step_time = step_time
                    last_cluster_id = cluster_id
                    last_expert_id = expert_id
                    last_active_expert = active_expert
                    last_seq_len = seq_len
                    last_tokens_per_step = tokens_per_step

                    metrics = {
                        "step": global_step,
                        "train/loss": loss_value,
                        "train/lr": current_lr,
                        "train/expert_id": expert_id,
                        "train/cluster_id": cluster_id,
                        "train/step_time": step_time,
                        "train/seq_len": seq_len,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_step": tokens_per_step,
                        "active_expert": active_expert,
                    }
                    write_jsonl(metrics_path, metrics)
                    if writer is not None:
                        writer.add_scalar("train/loss", loss_value, global_step)
                        writer.add_scalar("train/lr", current_lr, global_step)
                        writer.add_scalar("train/expert_id", expert_id, global_step)
                        writer.add_scalar("train/cluster_id", cluster_id, global_step)
                        writer.add_scalar("train/step_time", step_time, global_step)
                        writer.add_scalar("train/seq_len", seq_len, global_step)
                        writer.add_scalar("train/grad_norm", grad_norm, global_step)
                        writer.add_scalar("train/tokens_per_step", tokens_per_step, global_step)

                    should_log = global_step == 1 or args.logging_steps <= 1 or global_step % args.logging_steps == 0
                    if should_log:
                        if active_expert is not None:
                            print(
                                "TRAIN_STEP step={} expert={} cluster_id={} loss={:.6f} lr={:.6e} grad_norm={:.6f} seq_len={} tokens={} step_time={:.3f}s".format(
                                    global_step,
                                    active_expert,
                                    cluster_id,
                                    loss_value,
                                    current_lr,
                                    grad_norm,
                                    seq_len,
                                    tokens_per_step,
                                    step_time,
                                )
                            )
                        else:
                            print(
                                "TRAIN_STEP step={} cluster_id={} loss={:.6f} lr={:.6e} grad_norm={:.6f} seq_len={} tokens={} step_time={:.3f}s".format(
                                    global_step,
                                    cluster_id,
                                    loss_value,
                                    current_lr,
                                    grad_norm,
                                    seq_len,
                                    tokens_per_step,
                                    step_time,
                                )
                            )
                        if writer is not None:
                            writer.flush()

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_step_checkpoint(model, tokenizer, args, global_step)

                    if global_step >= args.max_steps:
                        break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if global_step > 0:
        print(
            "TRAIN_COMPLETE steps={} last_loss={:.6f} last_lr={:.6e} last_cluster_id={} last_expert_id={} last_step_time={:.3f}s".format(
                global_step,
                last_loss,
                last_lr,
                last_cluster_id,
                last_expert_id,
                last_step_time,
            )
        )
        if last_active_expert is not None:
            print("last active expert   : {}".format(last_active_expert))
        print("metrics jsonl        : {}".format(metrics_path))
        if tensorboard_log_dir is not None:
            print("tensorboard log dir  : {}".format(tensorboard_log_dir))

    summary = {
        "train_file": os.path.abspath(args.train_file),
        "output_dir": os.path.abspath(args.output_dir),
        "model_name_or_path": args.model_name_or_path,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "samples": len(samples),
        "batch_shape": list(first_batch["input_ids"].shape),
        "supervised_tokens_first_batch": supervised_tokens,
        "train_mode": active_mode,
        "num_experts": args.num_experts,
        "first_batch_cluster_id": first_batch_cluster_id,
        "tensorboard_log_dir": None if tensorboard_log_dir is None else os.path.abspath(tensorboard_log_dir),
        "metrics_path": os.path.abspath(metrics_path),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "max_steps": args.max_steps,
        "completed_steps": global_step,
        "last_loss": last_loss,
        "last_lr": last_lr,
        "last_grad_norm": last_grad_norm,
        "last_step_time": last_step_time,
        "last_cluster_id": last_cluster_id,
        "last_expert_id": last_expert_id,
        "last_seq_len": last_seq_len,
        "last_tokens_per_step": last_tokens_per_step,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = os.path.join(args.output_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_model:
        model.save_pretrained(os.path.join(args.output_dir, "model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))

    print("summary              : {}".format(summary_path))
    print("SINGLE_GPU_TRAIN_OK")


if __name__ == "__main__":
    main()
