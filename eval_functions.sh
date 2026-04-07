#!/bin/bash

# This script runs the entire evaluation of the fairseq wav2vec unsupervised pipeline

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_PATH=$DIR_PATH/$1 # the model should be a .pt file

# ==================== VITERBI DECODING ====================

# Run Viterbi decoding on a given split
run_viterbi_decoding() {
    local split=$1
    local output_dir=$2

    export HYDRA_FULL_ERROR=1
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
    export KENLM_ROOT="$KENLM_ROOT"
    export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

    mkdir -p "$output_dir"

    log "Running Viterbi decoding on $split set..."

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" \
        --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
        --config-name viterbi \
        fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
        fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
        fairseq.common_eval.path="$MODEL_PATH" \
        fairseq.task.text_data="$TEXT_OUTPUT/phones/" \
        fairseq.dataset.batch_size=1 \
        fairseq.dataset.num_workers=0 \
        fairseq.dataset.required_batch_size_multiple=1 \
        fairseq.dataset.gen_subset="$split" \
        results_path="$output_dir"

    if [ $? -eq 0 ]; then
        log "$split Viterbi decoding completed"
    else
        log "ERROR: $split Viterbi decoding failed"
        exit 1
    fi
}

# Legacy function kept for backwards compatibility
transcription_gans_viterbi() {
    run_viterbi_decoding "valid" "$GANS_OUTPUT_PHONES"
}

# ==================== EVALUATION ON ALL SPLITS ====================

evaluate_all_splits() {
    local step_name="evaluate_all_splits"

    log "Evaluating model on train, valid, and test splits..."

    mkdir -p "$EVAL_OUTPUT"

    # Decode each split
    for split in train valid test; do
        if [ -f "$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled/${split}.npy" ]; then
            run_viterbi_decoding "$split" "$EVAL_OUTPUT"
        else
            log "WARNING: No data for $split split, skipping"
        fi
    done

    log "All split evaluations completed"
}

# ==================== COMPUTE METRICS ====================

compute_split_metrics() {
    local split=$1
    local hyp_file="$EVAL_OUTPUT/${split}.txt"

    if [ ! -f "$hyp_file" ]; then
        log "WARNING: No predictions for $split ($hyp_file not found), skipping metrics"
        return
    fi

    # PER
    if [ -f "$TEST_REFS/${split}.phn" ]; then
        log "Computing PER for $split..."
        python "$DIR_PATH/compute_metrics.py" \
            --hyp "$hyp_file" \
            --ref "$TEST_REFS/${split}.phn" \
            --metric per \
            --detail \
            --output "$RESULTS_DIR/per_results_${split}.txt"
    else
        log "No phone references for $split, skipping PER"
    fi

    # WER
    if [ -f "$TEST_REFS/${split}.wrd" ]; then
        log "Computing WER for $split..."
        python "$DIR_PATH/compute_metrics.py" \
            --hyp "$hyp_file" \
            --ref "$TEST_REFS/${split}.wrd" \
            --metric wer \
            --detail \
            --output "$RESULTS_DIR/wer_results_${split}.txt"
    else
        log "No word references for $split, skipping WER"
    fi
}

compute_all_metrics() {
    log "Computing metrics for all splits..."

    for split in train valid test; do
        compute_split_metrics "$split"
    done

    # Print summary
    echo ""
    echo "=================================================================="
    echo "  EVALUATION RESULTS SUMMARY"
    echo "=================================================================="
    for split in train valid test; do
        per_file="$RESULTS_DIR/per_results_${split}.txt"
        wer_file="$RESULTS_DIR/wer_results_${split}.txt"
        if [ -f "$per_file" ]; then
            per_val=$(grep "Error Rate\|PER" "$per_file" | head -1)
            echo "  $split PER:  $per_val"
        fi
        if [ -f "$wer_file" ]; then
            wer_val=$(grep "Error Rate\|WER" "$wer_file" | head -1)
            echo "  $split WER:  $wer_val"
        fi
    done
    echo "=================================================================="
    echo ""
}

# ==================== GENERATE GRAPHS ====================

generate_graphs() {
    log "Generating training and evaluation graphs..."

    python "$DIR_PATH/generate_graphs.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$RESULTS_DIR/graphs"

    if [ $? -eq 0 ]; then
        log "Graphs generated in $RESULTS_DIR/graphs/"
    else
        log "WARNING: Graph generation failed (matplotlib may not be installed)"
    fi
}
