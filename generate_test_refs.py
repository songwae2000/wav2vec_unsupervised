#!/usr/bin/env python3
"""
Generate ground-truth phone and word references for evaluation.
Reads TSV manifests, finds matching LibriSpeech transcripts, and
produces .wrd (words) and .phn (phonemes via g2p_en) files.

Usage:
    python generate_test_refs.py --manifest-dir <dir> --librispeech-dir <dir> --output-dir <dir>
"""

import argparse
import glob
import os

from g2p_en import G2p


def load_transcripts(librispeech_dir):
    """Load all LibriSpeech transcripts into {utterance_id: text} dict."""
    transcripts = {}
    for trans_file in glob.glob(f"{librispeech_dir}/**/*.trans.txt", recursive=True):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
    return transcripts


def get_ids_from_tsv(tsv_path):
    """Extract utterance IDs from a fairseq-style TSV manifest."""
    ids = []
    with open(tsv_path) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        fname = line.strip().split("\t")[0]
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        ids.append(utt_id)
    return ids


def generate_refs_for_split(split, tsv_path, transcripts, g2p, output_dir):
    """Generate .wrd and .phn reference files for a given split."""
    test_ids = get_ids_from_tsv(tsv_path)
    print(f"  {split}: {len(test_ids)} utterances")

    wrd_lines = []
    phn_lines = []
    matched = 0

    for utt_id in test_ids:
        if utt_id in transcripts:
            text = transcripts[utt_id]
            matched += 1
        else:
            text = ""

        wrd_lines.append(text)

        # Phonemize
        words = text.strip().split()
        phones = []
        for w in words:
            phn = g2p(w)
            phn = [p[:-1] if p[-1].isnumeric() else p for p in phn]
            phones.extend([p for p in phn if p.strip() and p != " "])
        phn_lines.append(" ".join(phones))

    with open(os.path.join(output_dir, f"{split}.wrd"), "w") as f:
        f.write("\n".join(wrd_lines) + "\n")

    with open(os.path.join(output_dir, f"{split}.phn"), "w") as f:
        f.write("\n".join(phn_lines) + "\n")

    print(f"  {split}: matched {matched}/{len(test_ids)} transcripts")


def main():
    parser = argparse.ArgumentParser()
    # Support both old (single --test-tsv) and new (--manifest-dir) interfaces
    parser.add_argument("--test-tsv", default=None, help="(Legacy) Path to test.tsv manifest")
    parser.add_argument("--manifest-dir", default=None, help="Directory containing train.tsv, valid.tsv, test.tsv")
    parser.add_argument("--librispeech-dir", required=True, help="Path to LibriSpeech root")
    parser.add_argument("--output-dir", required=True, help="Output directory for refs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load transcripts
    transcripts = load_transcripts(args.librispeech_dir)
    print(f"  Loaded {len(transcripts)} transcripts from LibriSpeech")

    g2p = G2p()

    if args.manifest_dir:
        # Generate refs for all available splits
        for split in ["train", "valid", "test"]:
            tsv_path = os.path.join(args.manifest_dir, f"{split}.tsv")
            if os.path.exists(tsv_path):
                generate_refs_for_split(split, tsv_path, transcripts, g2p, args.output_dir)
            else:
                print(f"  {split}: no manifest found, skipping")
    elif args.test_tsv:
        # Legacy: single test file
        generate_refs_for_split("test", args.test_tsv, transcripts, g2p, args.output_dir)
    else:
        print("ERROR: Provide either --manifest-dir or --test-tsv")
        exit(1)

    print(f"  References saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
