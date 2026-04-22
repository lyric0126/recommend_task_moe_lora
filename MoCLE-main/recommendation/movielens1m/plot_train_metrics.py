#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip line {lineno}: {e}")
    if not rows:
        raise ValueError(f"No valid JSON rows found in {path}")
    return rows


def deduplicate_by_step(rows):
    step_to_row = {}
    for row in rows:
        if "step" not in row:
            continue
        try:
            step = int(row["step"])
        except Exception:
            continue
        step_to_row[step] = row
    dedup_rows = [step_to_row[k] for k in sorted(step_to_row.keys())]
    return dedup_rows


def extract_series(rows, key):
    xs, ys = [], []
    for row in rows:
        if "step" not in row or key not in row:
            continue
        step = row["step"]
        value = row[key]
        if value is None:
            continue
        try:
            xs.append(int(step))
            ys.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def moving_average(values, window):
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    if window <= 1:
        return values

    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def plot_metric(steps, values, title, ylabel, save_path, smooth_window=0, do_smooth=True, ylim=None):
    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, linewidth=0.6, alpha=0.25, label="raw")

    if do_smooth and smooth_window > 1 and len(values) > 1:
        smooth_values = moving_average(values, smooth_window)
        plt.plot(steps, smooth_values, linewidth=1.5, label=f"moving_avg_{smooth_window}")

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="x")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scatter(steps, values, title, ylabel, save_path, point_size=2):
    plt.figure(figsize=(12, 6))
    plt.scatter(steps, values, s=point_size, alpha=0.35)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hist(values, title, xlabel, save_path, bins=100):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_count_bar(values, title, xlabel, save_path):
    counter = Counter(int(v) for v in values)
    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]

    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_basic_stats(name, steps, values):
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))

    print(f"[STAT] {name} min    = {values[min_idx]:.6f} at step={steps[min_idx]}")
    print(f"[STAT] {name} max    = {values[max_idx]:.6f} at step={steps[max_idx]}")
    print(f"[STAT] {name} mean   = {np.mean(values):.6f}")
    print(f"[STAT] {name} median = {np.median(values):.6f}")
    print(f"[STAT] {name} last   = {values[-1]:.6f} at step={steps[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from JSONL.")
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to train_metrics.jsonl",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <jsonl_dir>/plots",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=200,
        help="Moving average window for smoothable metrics (default: 200)",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Do not deduplicate by step. By default, keep only the last record of each step.",
    )
    args = parser.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(os.path.dirname(jsonl_path), "plots")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    rows = load_jsonl(jsonl_path)
    raw_rows_count = len(rows)

    if args.no_dedup:
        used_rows = rows
        print(f"[INFO] loaded rows      : {raw_rows_count}")
        print(f"[INFO] dedup disabled   : using all rows")
    else:
        used_rows = deduplicate_by_step(rows)
        print(f"[INFO] loaded rows      : {raw_rows_count}")
        print(f"[INFO] dedup rows       : {len(used_rows)}")

    print(f"[INFO] output dir       : {outdir}")

    continuous_metrics = [
        ("train/loss", "loss"),
        ("train/grad_norm", "grad_norm"),
        ("train/seq_len", "seq_len"),
        ("train/step_time", "step_time"),
    ]

    discrete_metrics = [
        ("train/expert_id", "expert_id"),
        ("train/cluster_id", "cluster_id"),
    ]

    for key, name in continuous_metrics:
        steps, values = extract_series(used_rows, key)
        if len(steps) == 0:
            print(f"[WARN] no data found for {key}")
            continue

        print_basic_stats(name, steps, values)

        save_path = os.path.join(outdir, f"{name}.png")
        plot_metric(
            steps=steps,
            values=values,
            title=name,
            ylabel=name,
            save_path=save_path,
            smooth_window=args.smooth,
            do_smooth=True,
        )
        print(
            f"[OK] {name:12s} -> {save_path} "
            f"(points={len(values)}, first_step={steps[0]}, last_step={steps[-1]}, last_value={values[-1]:.6f})"
        )

        if name == "loss":
            zoom_path = os.path.join(outdir, "loss_zoom.png")
            plot_metric(
                steps=steps,
                values=values,
                title="loss_zoom",
                ylabel="loss",
                save_path=zoom_path,
                smooth_window=args.smooth,
                do_smooth=True,
                ylim=(0, 1.5),
            )
            print(f"[OK] loss_zoom    -> {zoom_path}")

            upper = float(np.percentile(values, 99))
            clip99_path = os.path.join(outdir, "loss_clip99.png")
            plot_metric(
                steps=steps,
                values=values,
                title="loss_clip99",
                ylabel="loss",
                save_path=clip99_path,
                smooth_window=args.smooth,
                do_smooth=True,
                ylim=(0, upper),
            )
            print(f"[OK] loss_clip99  -> {clip99_path} (ylim=0,{upper:.6f})")

            hist_path = os.path.join(outdir, "loss_hist.png")
            plot_hist(values, "loss_hist", "loss", hist_path, bins=100)
            print(f"[OK] loss_hist    -> {hist_path}")

    for key, name in discrete_metrics:
        steps, values = extract_series(used_rows, key)
        if len(steps) == 0:
            print(f"[WARN] no data found for {key}")
            continue

        print_basic_stats(name, steps, values)

        scatter_path = os.path.join(outdir, f"{name}_scatter.png")
        plot_scatter(
            steps=steps,
            values=values,
            title=f"{name}_scatter",
            ylabel=name,
            save_path=scatter_path,
            point_size=2,
        )
        print(f"[OK] {name+'_scatter':12s} -> {scatter_path}")

        hist_path = os.path.join(outdir, f"{name}_hist.png")
        plot_count_bar(
            values=values,
            title=f"{name}_hist",
            xlabel=name,
            save_path=hist_path,
        )
        print(f"[OK] {name+'_hist':12s} -> {hist_path}")

    print("[DONE] all plots saved.")


if __name__ == "__main__":
    main()