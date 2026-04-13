#!/bin/bash

# =============================================================================
# Download all required data and pretrained models for Wav2Vec-U pipeline
# =============================================================================
# Downloads:
#   1. LibriSpeech dev-clean (audio for train/val/test)
#   2. LibriSpeech LM text (unlabelled text for phoneme LM)
#   3. Pretrained wav2vec 2.0 model (wav2vec_vox_new.pt)
#   4. FastText language identification model (lid.176.bin)
# =============================================================================

set -e
set -o pipefail

source "$(dirname "$0")/utils.sh"

AUDIO_DIR="$DATA_ROOT/audio"
TEXT_DIR="$DATA_ROOT/text"
MODEL_DIR="$DATA_ROOT/models"

mkdir -p "$AUDIO_DIR" "$TEXT_DIR" "$MODEL_DIR"

# ==================== 1. DOWNLOAD LIBRISPEECH DEV-CLEAN ====================
download_librispeech() {
    local step_name="download_librispeech"

    if is_completed "$step_name"; then
        log "Skipping LibriSpeech download (already completed)"
        return 0
    fi

    log "Downloading LibriSpeech dev-clean..."
    mark_in_progress "$step_name"

    cd "$AUDIO_DIR"
    if [ ! -f "dev-clean.tar.gz" ]; then
        curl -L -O "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    fi

    if [ ! -d "LibriSpeech/dev-clean" ]; then
        log "Extracting dev-clean..."
        tar xzf dev-clean.tar.gz
    fi

    mark_completed "$step_name"
    log "LibriSpeech dev-clean downloaded and extracted"
}

# ==================== 2. PREPARE AUDIO SPLITS ====================
prepare_audio_splits() {
    local step_name="prepare_audio_splits"

    if is_completed "$step_name"; then
        log "Skipping audio split preparation (already completed)"
        return 0
    fi

    log "Preparing train/val/test audio splits from LibriSpeech..."
    mark_in_progress "$step_name"

    mkdir -p "$AUDIO_DIR/train_wav" "$AUDIO_DIR/val_wav" "$AUDIO_DIR/test_wav"

    activate_venv

    python3 -c "
import os, glob, random, soundfile as sf

random.seed(42)
libri_dir = '$AUDIO_DIR/LibriSpeech/dev-clean'
train_dir = '$AUDIO_DIR/train_wav'
val_dir = '$AUDIO_DIR/val_wav'
test_dir = '$AUDIO_DIR/test_wav'

# Skip if already populated
if len(os.listdir(train_dir)) > 0:
    print('Audio splits already exist, skipping')
    exit(0)

# Get all flac files
flac_files = sorted(glob.glob(f'{libri_dir}/**/*.flac', recursive=True))
print(f'Found {len(flac_files)} flac files')

# Shuffle and split: 80% train, 10% val, 10% test
random.shuffle(flac_files)
n = len(flac_files)
n_val = max(int(n * 0.10), 20)
n_test = max(int(n * 0.10), 20)
n_train = n - n_val - n_test

train_files = flac_files[:n_train]
val_files = flac_files[n_train:n_train + n_val]
test_files = flac_files[n_train + n_val:]

print(f'Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')

def convert_files(files, out_dir, label):
    for i, f in enumerate(files):
        wav_name = os.path.basename(f).replace('.flac', '.wav')
        data, sr = sf.read(f)
        sf.write(os.path.join(out_dir, wav_name), data, sr, subtype='PCM_16')
        if (i+1) % 500 == 0:
            print(f'  {label}: {i+1}/{len(files)}')
    print(f'  {label}: {len(files)} files converted')

convert_files(train_files, train_dir, 'train')
convert_files(val_files, val_dir, 'val')
convert_files(test_files, test_dir, 'test')
"

    mark_completed "$step_name"
    log "Audio splits prepared"
}

# ==================== 3. DOWNLOAD LM TEXT ====================
download_lm_text() {
    local step_name="download_lm_text"

    if is_completed "$step_name"; then
        log "Skipping LM text download (already completed)"
        return 0
    fi

    log "Preparing LM text data..."
    mark_in_progress "$step_name"

    if [ ! -f "$TEXT_DIR/lm_text_50k.txt" ]; then
        # Use LibriSpeech LM corpus or generate from transcripts
        log "Extracting text from LibriSpeech transcripts..."
        find "$AUDIO_DIR/LibriSpeech" -name "*.trans.txt" -exec cat {} \; | \
            cut -d' ' -f2- | head -50000 > "$TEXT_DIR/lm_text_50k.txt"
    fi

    log "LM text lines: $(wc -l < "$TEXT_DIR/lm_text_50k.txt")"
    mark_completed "$step_name"
    log "LM text prepared"
}

# ==================== 4. DOWNLOAD PRETRAINED MODELS ====================
download_pretrained_models() {
    local step_name="download_pretrained_models"

    if is_completed "$step_name"; then
        log "Skipping pretrained model download (already completed)"
        return 0
    fi

    log "Downloading pretrained models..."
    mark_in_progress "$step_name"

    cd "$MODEL_DIR"

    # wav2vec 2.0 pretrained model
    if [ ! -f "wav2vec_vox_new.pt" ]; then
        log "Downloading wav2vec_vox_new.pt (~1.2GB)..."
        wget -q --show-progress "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt"
    else
        log "wav2vec_vox_new.pt already exists"
    fi

    # FastText language identification model
    if [ ! -f "lid.176.bin" ]; then
        log "Downloading lid.176.bin..."
        wget -q --show-progress "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    else
        log "lid.176.bin already exists"
    fi

    mark_completed "$step_name"
    log "Pretrained models downloaded"
}

# ==================== RUN ALL DOWNLOADS ====================
create_dirs
download_librispeech
prepare_audio_splits
download_lm_text
download_pretrained_models

log "All data downloaded and prepared!"
echo ""
echo "  Train audio: $AUDIO_DIR/train_wav ($(ls "$AUDIO_DIR/train_wav" 2>/dev/null | wc -l) files)"
echo "  Val audio:   $AUDIO_DIR/val_wav ($(ls "$AUDIO_DIR/val_wav" 2>/dev/null | wc -l) files)"
echo "  Test audio:  $AUDIO_DIR/test_wav ($(ls "$AUDIO_DIR/test_wav" 2>/dev/null | wc -l) files)"
echo "  LM text:     $TEXT_DIR/lm_text_50k.txt"
echo "  Models:      $MODEL_DIR/"
echo ""
