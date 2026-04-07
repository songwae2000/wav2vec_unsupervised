#!/bin/bash

# Source the function definitions
source "$(dirname "$0")/wav2vec_functions.sh"

create_dirs #creates directories for storing outputs from the different steps

activate_venv
setup_path  #add kenlm and kaldi to the LD_LIBRARY directory

log "Starting wav2vec unsupervised pipeline for $DATASET_NAME"

log "It creates a manifest files for the audio dataset audio format"

create_manifests_train 0
create_manifests_val 0
create_manifests_test 0

# Subsample raw manifests EARLY so VAD only processes the small subset
subsample_manifests

# Clean stale data from previous runs so only the subsampled set gets processed
rm -rf "$NONSIL_AUDIO/train" "$NONSIL_AUDIO/val" "$NONSIL_AUDIO/test" 2>/dev/null
rm -f "$MANIFEST_NONSIL_DIR/train.tsv" "$MANIFEST_NONSIL_DIR/valid.tsv" "$MANIFEST_NONSIL_DIR/test.tsv" 2>/dev/null
# Reset checkpoints for steps whose outputs we just deleted
for step in create_rVADfast remove_silence create_manifests_nonsil_train create_manifests_nonsil_val create_manifests_nonsil_test generate_test_refs prepare_audio; do
    sed -i '' "/^${step}:/d" "$CHECKPOINT_FILE" 2>/dev/null
done
log "Cleaned processed audio and nonsil manifests for fresh run"

#creates new manifest with silence removed

create_rVADfast # identifies the sequence of silence in an audio (train + val + test)
remove_silence # removes the silence sequence found by rvad in the audio (train + val + test)
create_manifests_nonsil_train 0.1
create_manifests_nonsil_val 0.1
create_manifests_nonsil_test  # create nonsil manifest for test set

# Also clean clustering dir for fresh audio processing
rm -rf "$CLUSTERING_DIR" 2>/dev/null
mkdir -p "$CLUSTERING_DIR"

# Generate ground-truth phoneme references for all splits
generate_test_refs

# Audio and text processing
#     prepare_audio:  processes the unlabelled audio (features, clustering, PCA)
#     prepare_text:   processes the unlabelled text (phonemization, KenLM)

prepare_audio
prepare_text

log "Pipeline completed successfully!"
