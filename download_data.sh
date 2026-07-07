#!/usr/bin/env bash
# Download VoiceBank+DEMAND (28-speaker version) from Edinburgh DataShare
# Usage: bash download_data.sh

set -e

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

BASE_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"

declare -A FILES=(
  ["clean_trainset_28spk_wav.zip"]="$BASE_URL/clean_trainset_28spk_wav.zip"
  ["noisy_trainset_28spk_wav.zip"]="$BASE_URL/noisy_trainset_28spk_wav.zip"
  ["clean_testset_wav.zip"]="$BASE_URL/clean_testset_wav.zip"
  ["noisy_testset_wav.zip"]="$BASE_URL/noisy_testset_wav.zip"
)

for fname in "${!FILES[@]}"; do
  url="${FILES[$fname]}"
  if [ -f "$fname" ]; then
    echo "Already downloaded: $fname"
  else
    echo "Downloading $fname ..."
    wget -c "$url" -O "$fname"
  fi
done

echo "Extracting..."
for fname in "${!FILES[@]}"; do
  dirname="${fname%.zip}"
  if [ -d "$dirname" ]; then
    echo "Already extracted: $dirname"
  else
    unzip -q "$fname" -d .
  fi
done

echo "Done. Files are in $RAW_DIR/"
ls -la