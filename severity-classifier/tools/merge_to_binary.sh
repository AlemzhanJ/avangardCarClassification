#!/usr/bin/env bash
set -euo pipefail

# Merge undamaged_dataset.folder (-> undamaged) and damaged_dataset (-> damaged)
# into a binary ImageFolder dataset at severity-classifier/binary_dataset with splits train/valid/test.

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNDAMAGED_SRC="$BASE_DIR/undamaged_dataset.folder"
DAMAGED_SRC="$BASE_DIR/damaged_dataset"
OUT="$BASE_DIR/binary_dataset"

echo "Undamaged src: $UNDAMAGED_SRC"
echo "Damaged src:   $DAMAGED_SRC"
echo "Output dir:    $OUT"

if [[ ! -d "$UNDAMAGED_SRC" ]]; then
  echo "ERROR: undamaged source not found: $UNDAMAGED_SRC" >&2
  exit 1
fi
if [[ ! -d "$DAMAGED_SRC" ]]; then
  echo "ERROR: damaged source not found: $DAMAGED_SRC" >&2
  exit 1
fi

shopt -s nullglob

copy_images() {
  local src_dir="$1" dst_dir="$2" prefix="$3"
  mkdir -p "$dst_dir"
  # Find common image extensions (case-insensitive)
  while IFS= read -r -d '' f; do
    local bn
    bn="$(basename "$f")"
    local dst="$dst_dir/${prefix}__${bn}"
    # If collision, add a random suffix
    if [[ -e "$dst" ]]; then
      dst="$dst_dir/${prefix}__${RANDOM}_${bn}"
    fi
    cp "$f" "$dst"
  done < <(find "$src_dir" -type f \( \
      -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \
    \) -print0)
}

for split in train valid test; do
  echo "\n== Processing split: $split"
  mkdir -p "$OUT/$split/undamaged" "$OUT/$split/damaged"

  # UNDAMAGED
  if [[ -d "$UNDAMAGED_SRC/$split" ]]; then
    echo "  + From undamaged_dataset.folder -> undamaged"
    copy_images "$UNDAMAGED_SRC/$split" "$OUT/$split/undamaged" "undamagedDS"
  else
    echo "  ! Skip: missing $UNDAMAGED_SRC/$split"
  fi

  # DAMAGED
  if [[ -d "$DAMAGED_SRC/$split" ]]; then
    echo "  + From damaged_dataset -> damaged"
    copy_images "$DAMAGED_SRC/$split" "$OUT/$split/damaged" "damagedDS"
  else
    echo "  ! Skip: missing $DAMAGED_SRC/$split"
  fi
done

echo "\nDone. Sample counts:"
for split in train valid test; do
  for cls in undamaged damaged; do
    if [[ -d "$OUT/$split/$cls" ]]; then
      cnt=$(find "$OUT/$split/$cls" -type f | wc -l | tr -d ' ')
      echo "  $split/$cls: $cnt files"
    fi
  done
done

