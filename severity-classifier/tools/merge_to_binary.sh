#!/usr/bin/env bash
set -euo pipefail

# Merge undamaged_dataset.folder (all classes -> undamaged)
# and damaged_dataset (all classes -> damaged)
# into a binary ImageFolder dataset at severity-classifier/binary_dataset
# with splits train/valid/test.

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNDAMAGED_SRC="$BASE_DIR/undamaged_dataset.folder"
DAMAGED_SRC="$BASE_DIR/damaged_dataset"
OUT="$BASE_DIR/binary_dataset"

echo "Undamaged src: $UNDAMAGED_SRC"
echo "Damaged src:   $DAMAGED_SRC"
echo "Output dir:    $OUT"

shopt -s nullglob

sanitize() {
  # Replace spaces and slashes for safe filenames
  echo "$1" | sed -e 's/[[:space:]]\+/_/g' -e 's#/#_#g'
}

copy_images() {
  local src_dir="$1" dst_dir="$2" prefix="$3"
  mkdir -p "$dst_dir"
  while IFS= read -r -d '' f; do
    local bn
    bn="$(basename "$f")"
    local dst="$dst_dir/${prefix}__${bn}"
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

  # 1) UNDAMAGED DATASET -> undamaged (all subfolders)
  if [[ -d "$UNDAMAGED_SRC/$split" ]]; then
    echo "  + From undamaged_dataset.folder -> undamaged"
    for d in "$UNDAMAGED_SRC/$split"/*; do
      [[ -d "$d" ]] || continue
      brand="$(sanitize "$(basename "$d")")"
      copy_images "$d" "$OUT/$split/undamaged" "undmg_${brand}"
    done
  fi

  # 2) DAMAGED DATASET -> damaged (all subfolders)
  if [[ -d "$DAMAGED_SRC/$split" ]]; then
    echo "  + From damaged_dataset -> damaged"
    for d in "$DAMAGED_SRC/$split"/*; do
      [[ -d "$d" ]] || continue
      cname="$(sanitize "$(basename "$d")")"
      copy_images "$d" "$OUT/$split/damaged" "dmg_${cname}"
    done
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

