#!/bin/bash

# Source the function definitions
source "$(dirname "$0")/eval_functions.sh"

create_dirs
activate_venv

# ==================== EVALUATION PIPELINE ====================
# 1. Run Viterbi decoding on train, valid, and test splits
# 2. Compute PER and WER metrics for each split
# 3. Generate training curves and evaluation graphs

evaluate_all_splits    # Viterbi decode all splits
compute_all_metrics    # PER/WER for each split
generate_graphs        # Training curves + metric bar charts

log "Evaluation pipeline completed!"
