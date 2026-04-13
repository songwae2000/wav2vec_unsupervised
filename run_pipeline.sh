#!/bin/bash

# =============================================================================
# Wav2Vec-U Complete End-to-End Pipeline
# =============================================================================
# Runs the FULL unsupervised speech recognition pipeline:
#
#   Phase 1: Download data & pretrained models     (download_data.sh)
#   Phase 2: Data & Audio/Text Preparation         (run_wav2vec.sh)
#   Phase 3: GAN Training                          (run_gans.sh)
#   Phase 4: Evaluation, Metrics & Graphs          (run_eval.sh)
#
# Usage:
#   ./run_pipeline.sh [checkpoint_path]
#
# Configuration:
#   Edit utils.sh to change NUM_TRAIN, NUM_VAL, NUM_TEST, MAX_UPDATES, etc.
#   All data paths are auto-detected from the script directory.
# =============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Wav2Vec-U Complete End-to-End Pipeline                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Project dir:   $DIR_PATH"
echo "║  Fairseq:       $FAIRSEQ_ROOT"
echo "║  Train samples: $NUM_TRAIN"
echo "║  Val samples:   $NUM_VAL"
echo "║  Test samples:  $NUM_TEST"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ==================== PHASE 1: DOWNLOAD DATA ====================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Download Data & Pretrained Models                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

bash "$SCRIPT_DIR/download_data.sh"

# ==================== PHASE 2: DATA & AUDIO/TEXT PREPARATION ====================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Data Preparation + Audio/Text Processing              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# run_wav2vec.sh expects: <train_dir> <val_dir> <test_dir> <text_file>
bash "$SCRIPT_DIR/run_wav2vec.sh" \
    "$DATA_ROOT/audio/train_wav" \
    "$DATA_ROOT/audio/val_wav" \
    "$DATA_ROOT/audio/test_wav" \
    "$DATA_ROOT/text/lm_text_50k.txt"

# ==================== PHASE 3: GAN TRAINING ====================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: GAN Training                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

bash "$SCRIPT_DIR/run_gans.sh"

# ==================== PHASE 4: EVALUATION, METRICS & GRAPHS ====================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 4: Evaluation, Metrics & Graphs                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Use provided checkpoint or default to checkpoint_best.pt from training
CHECKPOINT=${1:-"data/results/librispeech/checkpoint_best.pt"}
bash "$SCRIPT_DIR/run_eval.sh" "$CHECKPOINT"

# ==================== PIPELINE COMPLETE ====================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE COMPLETE                             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  Outputs:                                                        ║"
echo "║    Checkpoints:    $RESULTS_DIR/"
echo "║    Training log:   $RESULTS_DIR/training.log"
echo "║    Predictions:    $EVAL_OUTPUT/"
echo "║    Graphs:         $RESULTS_DIR/graphs/"
echo "║                                                                  ║"
echo "║  Metrics:                                                        ║"
for split in train valid test; do
    per_file="$RESULTS_DIR/per_results_${split}.txt"
    if [ -f "$per_file" ]; then
        per_val=$(grep "Error Rate\|PER" "$per_file" | head -1)
        printf "║    %-8s PER: %s\n" "$split" "$per_val"
    fi
done
for split in train valid test; do
    wer_file="$RESULTS_DIR/wer_results_${split}.txt"
    if [ -f "$wer_file" ]; then
        wer_val=$(grep "Error Rate\|WER" "$wer_file" | head -1)
        printf "║    %-8s WER: %s\n" "$split" "$wer_val"
    fi
done
echo "║                                                                  ║"
echo "║  Graphs:                                                         ║"
for graph in "$RESULTS_DIR/graphs/"*.png; do
    if [ -f "$graph" ]; then
        echo "║    $(basename "$graph")"
    fi
done
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
