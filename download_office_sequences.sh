#!/usr/bin/env bash
# Downloads sequence.zip for 105 office scenes and extracts RGB frames.
#
# Usage: bash scan2therm/download_office_sequences.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="$SCRIPT_DIR/office_scenes_105.txt"
DATA_DIR="$PROJECT_DIR/3DSSG/data/3RScan/data/3RScan"
BASE_URL="http://campar.in.tum.de/public_datasets/3RScan/Dataset"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "ERROR: Scene list not found: $SCENE_LIST"
    exit 1
fi

mapfile -t SCAN_IDS < "$SCENE_LIST"
TOTAL=${#SCAN_IDS[@]}
echo "Found $TOTAL scans to process."

for i in "${!SCAN_IDS[@]}"; do
    scan_id="${SCAN_IDS[$i]}"
    idx=$((i + 1))
    scan_dir="$DATA_DIR/$scan_id"
    zip_file="$scan_dir/sequence.zip"
    seq_dir="$scan_dir/sequence"

    # Skip if already extracted
    if [[ -d "$seq_dir" ]]; then
        echo "[$idx/$TOTAL] Skipping $scan_id (sequence/ already exists)"
        continue
    fi

    # Download if needed
    if [[ ! -f "$zip_file" ]]; then
        echo "[$idx/$TOTAL] Downloading $scan_id ..."
        mkdir -p "$scan_dir"
        wget -q --show-progress -O "$zip_file" "$BASE_URL/$scan_id/sequence.zip" || {
            echo "  ERROR: Download failed for $scan_id, removing partial file"
            rm -f "$zip_file"
            continue
        }
    else
        echo "[$idx/$TOTAL] $scan_id sequence.zip already downloaded"
    fi

    # Extract
    echo "  Extracting $scan_id ..."
    unzip -q "$zip_file" -d "$seq_dir" || {
        echo "  ERROR: Extraction failed for $scan_id"
        continue
    }
done

echo "Done. Processed $TOTAL scans."
