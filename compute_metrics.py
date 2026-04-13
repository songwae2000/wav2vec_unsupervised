#!/usr/bin/env python3
"""
Compute Phone Error Rate (PER) and Word Error Rate (WER) for Wav2Vec-U evaluation.

Usage:
    python compute_metrics.py --hyp <hyp_file> --ref <ref_file> [--metric per|wer|both]
"""

import argparse
import editdistance


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def compute_error_rate(hyps, refs):
    """Compute token-level error rate (works for both PER and WER)."""
    total_edits = 0
    total_ref_len = 0

    details = []
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()

        edits = editdistance.eval(hyp_tokens, ref_tokens)
        total_edits += edits
        total_ref_len += len(ref_tokens)

        sample_er = edits / max(len(ref_tokens), 1) * 100
        details.append({
            "idx": i,
            "hyp": hyp,
            "ref": ref,
            "edits": edits,
            "ref_len": len(ref_tokens),
            "error_rate": sample_er,
        })

    overall_er = total_edits / max(total_ref_len, 1) * 100
    return overall_er, total_edits, total_ref_len, details


def main():
    parser = argparse.ArgumentParser(description="Compute PER/WER metrics")
    parser.add_argument("--hyp", required=True, help="Hypothesis file (one per line)")
    parser.add_argument("--ref", required=True, help="Reference file (one per line)")
    parser.add_argument("--metric", default="both", choices=["per", "wer", "both"],
                        help="Which metric to report")
    parser.add_argument("--detail", action="store_true", help="Print per-sample details")
    parser.add_argument("--output", default=None, help="Save results to file")
    args = parser.parse_args()

    hyps = load_lines(args.hyp)
    refs = load_lines(args.ref)

    if len(hyps) != len(refs):
        print(f"WARNING: hyp has {len(hyps)} lines, ref has {len(refs)} lines. Using min.")
        n = min(len(hyps), len(refs))
        hyps = hyps[:n]
        refs = refs[:n]

    error_rate, total_edits, total_ref_len, details = compute_error_rate(hyps, refs)
    metric_name = "PER" if args.metric == "per" else "WER" if args.metric == "wer" else "Error Rate"

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  {metric_name} Results")
    print(f"{'=' * 60}")
    print(f"  Samples:      {len(hyps)}")
    print(f"  Total edits:  {total_edits}")
    print(f"  Total tokens: {total_ref_len}")
    print(f"  {metric_name}:       {error_rate:.2f}%")
    print(f"{'=' * 60}\n")

    if args.detail:
        print(f"{'Idx':>4} | {'ER':>7} | {'Edits':>6} | {'RefLen':>6} | Ref -> Hyp")
        print("-" * 80)
        for d in details:
            print(f"{d['idx']:>4} | {d['error_rate']:>6.1f}% | {d['edits']:>6} | {d['ref_len']:>6} | "
                  f"{d['ref'][:30]} -> {d['hyp'][:30]}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"{metric_name}: {error_rate:.2f}%\n")
            f.write(f"Samples: {len(hyps)}\n")
            f.write(f"Total edits: {total_edits}\n")
            f.write(f"Total ref tokens: {total_ref_len}\n")
            f.write(f"\nPer-sample details:\n")
            for d in details:
                f.write(f"  [{d['idx']}] ER={d['error_rate']:.1f}% "
                        f"ref='{d['ref']}' hyp='{d['hyp']}'\n")
        print(f"Results saved to {args.output}")

    return error_rate


if __name__ == "__main__":
    main()
