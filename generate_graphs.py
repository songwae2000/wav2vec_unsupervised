#!/usr/bin/env python3
"""
Generate training and evaluation graphs for Wav2Vec-U pipeline.

Produces:
  1. Training loss curve (Generator vs Discriminator)
  2. Code perplexity over training
  3. Gradient norm over training
  4. Validation metrics over training
  5. PER/WER bar chart across splits (train/val/test)

Usage:
    python generate_graphs.py --results-dir <path> --output-dir <path>
"""

import argparse
import json
import os
import re


def parse_training_log(log_path):
    """Parse fairseq training log to extract metrics per epoch."""
    train_metrics = []
    valid_metrics = []

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # Match train log lines
            if "[train][INFO]" in line and '"epoch"' in line:
                try:
                    json_str = line.split("[INFO] - ")[1].strip()
                    data = json.loads(json_str)
                    train_metrics.append(data)
                except (IndexError, json.JSONDecodeError):
                    continue

            # Match valid log lines
            if "[valid][INFO]" in line and '"epoch"' in line:
                try:
                    json_str = line.split("[INFO] - ")[1].strip()
                    data = json.loads(json_str)
                    valid_metrics.append(data)
                except (IndexError, json.JSONDecodeError):
                    continue

    return train_metrics, valid_metrics


def safe_float(val):
    """Convert value to float, returning None for null/None."""
    if val is None or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def plot_training_losses(train_metrics, output_dir):
    """Plot generator and discriminator losses over training."""
    import matplotlib.pyplot as plt

    updates = [m.get("train_num_updates", i) for i, m in enumerate(train_metrics)]
    total_loss = [safe_float(m.get("train_loss")) for m in train_metrics]
    gen_loss = [safe_float(m.get("train_loss_dense_g")) for m in train_metrics]
    disc_loss = [safe_float(m.get("train_loss_dense_d")) for m in train_metrics]
    smoothness = [safe_float(m.get("train_loss_smoothness")) for m in train_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Wav2Vec-U GAN Training Curves", fontsize=16, fontweight="bold")

    # Total loss
    ax = axes[0, 0]
    vals = [(u, v) for u, v in zip(updates, total_loss) if v is not None]
    if vals:
        ax.plot([v[0] for v in vals], [v[1] for v in vals], "b-", linewidth=0.8)
    ax.set_title("Total Training Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Generator vs Discriminator
    ax = axes[0, 1]
    gen_vals = [(u, v) for u, v in zip(updates, gen_loss) if v is not None]
    disc_vals = [(u, v) for u, v in zip(updates, disc_loss) if v is not None]
    if gen_vals:
        ax.plot([v[0] for v in gen_vals], [v[1] for v in gen_vals], "g-", label="Generator", linewidth=0.8)
    if disc_vals:
        ax.plot([v[0] for v in disc_vals], [v[1] for v in disc_vals], "r-", label="Discriminator", linewidth=0.8)
    ax.set_title("Generator vs Discriminator Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Smoothness loss
    ax = axes[1, 0]
    sm_vals = [(u, v) for u, v in zip(updates, smoothness) if v is not None]
    if sm_vals:
        ax.plot([v[0] for v in sm_vals], [v[1] for v in sm_vals], "m-", linewidth=0.8)
    ax.set_title("Smoothness Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    gnorm = [safe_float(m.get("train_gnorm")) for m in train_metrics]
    gn_vals = [(u, v) for u, v in zip(updates, gnorm) if v is not None]
    if gn_vals:
        ax.plot([v[0] for v in gn_vals], [v[1] for v in gn_vals], "orange", linewidth=0.8)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_code_perplexity(train_metrics, output_dir):
    """Plot code perplexity and temperature over training."""
    import matplotlib.pyplot as plt

    updates = [m.get("train_num_updates", i) for i, m in enumerate(train_metrics)]
    code_ppl = [safe_float(m.get("train_code_ppl")) for m in train_metrics]
    temp = [safe_float(m.get("train_temp")) for m in train_metrics]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle("Code Perplexity & Temperature", fontsize=14, fontweight="bold")

    ppl_vals = [(u, v) for u, v in zip(updates, code_ppl) if v is not None]
    if ppl_vals:
        ax1.plot([v[0] for v in ppl_vals], [v[1] for v in ppl_vals], "b-", label="Code PPL", linewidth=0.8)
    ax1.set_xlabel("Updates")
    ax1.set_ylabel("Code Perplexity", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    temp_vals = [(u, v) for u, v in zip(updates, temp) if v is not None]
    if temp_vals:
        ax2.plot([v[0] for v in temp_vals], [v[1] for v in temp_vals], "r--", label="Temperature", linewidth=0.8)
    ax2.set_ylabel("Temperature", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "code_perplexity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_validation_metrics(valid_metrics, output_dir):
    """Plot validation LM perplexity and vocab coverage over training."""
    import matplotlib.pyplot as plt

    if not valid_metrics:
        print("  No validation metrics found, skipping validation plot")
        return

    updates = [m.get("valid_num_updates", i) for i, m in enumerate(valid_metrics)]
    lm_ppl = [safe_float(m.get("valid_lm_ppl")) for m in valid_metrics]
    weighted_ppl = [safe_float(m.get("valid_weighted_lm_ppl")) for m in valid_metrics]
    vocab_pct = [safe_float(m.get("valid_vocab_seen_pct")) for m in valid_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Validation Metrics Over Training", fontsize=14, fontweight="bold")

    # LM PPL
    ax = axes[0]
    ppl_vals = [(u, v) for u, v in zip(updates, lm_ppl) if v is not None and v < 1e6]
    wppl_vals = [(u, v) for u, v in zip(updates, weighted_ppl) if v is not None and v < 1e6]
    if ppl_vals:
        ax.plot([v[0] for v in ppl_vals], [v[1] for v in ppl_vals], "b-o", label="LM PPL", markersize=4)
    if wppl_vals:
        ax.plot([v[0] for v in wppl_vals], [v[1] for v in wppl_vals], "r-s", label="Weighted LM PPL", markersize=4)
    ax.set_title("Language Model Perplexity")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vocab coverage
    ax = axes[1]
    voc_vals = [(u, v) for u, v in zip(updates, vocab_pct) if v is not None]
    if voc_vals:
        ax.plot([v[0] for v in voc_vals], [v[1] * 100 for v in voc_vals], "g-o", markersize=4)
    ax.set_title("Vocabulary Coverage")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Vocab Seen (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "validation_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_error_rates(results_dir, output_dir):
    """Plot PER and WER bar chart for each split."""
    import matplotlib.pyplot as plt

    splits = ["train", "valid", "test"]
    per_values = {}
    wer_values = {}

    for split in splits:
        per_file = os.path.join(results_dir, f"per_results_{split}.txt")
        wer_file = os.path.join(results_dir, f"wer_results_{split}.txt")

        # Also check non-split naming for test
        if split == "test":
            if not os.path.exists(per_file):
                per_file = os.path.join(results_dir, "per_results.txt")
            if not os.path.exists(wer_file):
                wer_file = os.path.join(results_dir, "wer_results.txt")

        if os.path.exists(per_file):
            with open(per_file) as f:
                for line in f:
                    match = re.search(r"(?:PER|Error Rate):\s*([\d.]+)%", line)
                    if match:
                        per_values[split] = float(match.group(1))
                        break

        if os.path.exists(wer_file):
            with open(wer_file) as f:
                for line in f:
                    match = re.search(r"(?:WER|Error Rate):\s*([\d.]+)%", line)
                    if match:
                        wer_values[split] = float(match.group(1))
                        break

    if not per_values and not wer_values:
        print("  No error rate results found, skipping bar chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Metrics by Split", fontsize=14, fontweight="bold")

    colors = {"train": "#2196F3", "valid": "#FF9800", "test": "#4CAF50"}

    # PER bar chart
    ax = axes[0]
    if per_values:
        bars = ax.bar(per_values.keys(), per_values.values(),
                      color=[colors.get(k, "gray") for k in per_values.keys()],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, per_values.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Phone Error Rate (PER)")
    ax.set_ylabel("PER (%)")
    ax.set_ylim(0, min(max(per_values.values()) * 1.2, 120) if per_values else 100)
    ax.grid(True, alpha=0.3, axis="y")

    # WER bar chart
    ax = axes[1]
    if wer_values:
        bars = ax.bar(wer_values.keys(), wer_values.values(),
                      color=[colors.get(k, "gray") for k in wer_values.keys()],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, wer_values.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Word Error Rate (WER)")
    ax.set_ylabel("WER (%)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "error_rates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_training_summary(train_metrics, valid_metrics, output_dir):
    """Single-page summary combining key metrics."""
    import matplotlib.pyplot as plt

    updates = [m.get("train_num_updates", i) for i, m in enumerate(train_metrics)]
    total_loss = [safe_float(m.get("train_loss")) for m in train_metrics]
    code_ppl = [safe_float(m.get("train_code_ppl")) for m in train_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Training Summary", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    vals = [(u, v) for u, v in zip(updates, total_loss) if v is not None]
    if vals:
        ax.plot([v[0] for v in vals], [v[1] for v in vals], "b-", linewidth=0.8)
    ax.set_title("Training Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Code PPL
    ax = axes[1]
    ppl_vals = [(u, v) for u, v in zip(updates, code_ppl) if v is not None]
    if ppl_vals:
        ax.plot([v[0] for v in ppl_vals], [v[1] for v in ppl_vals], "g-", linewidth=0.8)
    ax.set_title("Code Perplexity")
    ax.set_xlabel("Updates")
    ax.set_ylabel("PPL")
    ax.grid(True, alpha=0.3)

    # Validation LM PPL
    ax = axes[2]
    if valid_metrics:
        v_updates = [m.get("valid_num_updates", i) for i, m in enumerate(valid_metrics)]
        v_ppl = [safe_float(m.get("valid_lm_ppl")) for m in valid_metrics]
        vppl_vals = [(u, v) for u, v in zip(v_updates, v_ppl) if v is not None and v < 1e6]
        if vppl_vals:
            ax.plot([v[0] for v in vppl_vals], [v[1] for v in vppl_vals], "r-o", markersize=4)
    ax.set_title("Validation LM Perplexity")
    ax.set_xlabel("Updates")
    ax.set_ylabel("PPL")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Wav2Vec-U training and evaluation graphs")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for graphs (default: results-dir/graphs)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating graphs...")
    print(f"  Results: {args.results_dir}")
    print(f"  Output:  {output_dir}\n")

    # Parse training log
    log_path = os.path.join(args.results_dir, "training.log")
    if os.path.exists(log_path):
        train_metrics, valid_metrics = parse_training_log(log_path)
        print(f"  Parsed {len(train_metrics)} train epochs, {len(valid_metrics)} valid checkpoints")

        if train_metrics:
            plot_training_losses(train_metrics, output_dir)
            plot_code_perplexity(train_metrics, output_dir)
            plot_training_summary(train_metrics, valid_metrics, output_dir)

        if valid_metrics:
            plot_validation_metrics(valid_metrics, output_dir)
    else:
        print(f"  WARNING: No training log found at {log_path}")

    # Plot error rates
    plot_error_rates(args.results_dir, output_dir)

    print(f"\nAll graphs saved to {output_dir}/")


if __name__ == "__main__":
    main()
