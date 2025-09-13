#!/usr/bin/env bash
set -euo pipefail

# Merge clean_dataset (all classes -> clean) and ditry_dataset (dirt* -> dirty, clean -> clean)
# into a binary ImageFolder dataset at dirty-classifier/binary_dataset with splits train/valid/test.

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLEAN_SRC="$BASE_DIR/clean_dataset"
DIRTY_SRC="$BASE_DIR/ditry_dataset"
OUT="$BASE_DIR/binary_dataset"

echo "Clean src:  $CLEAN_SRC"
echo "Dirty src:  $DIRTY_SRC"
echo "Output dir: $OUT"

shopt -s nullglob

sanitize() {
  # Replace spaces and slashes for safe filenames
  echo "$1" | sed -e 's/[[:space:]]\+/_/g' -e 's#/#_#g'
}

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
  mkdir -p "$OUT/$split/clean" "$OUT/$split/dirty"

  # 1) CLEAN DATASET -> clean
  if [[ -d "$CLEAN_SRC/$split" ]]; then
    echo "  + From clean_dataset -> clean"
    for d in "$CLEAN_SRC/$split"/*; do
      [[ -d "$d" ]] || continue
      brand="$(sanitize "$(basename "$d")")"
      copy_images "$d" "$OUT/$split/clean" "cleanDS_${brand}"
    done
  fi

  # 2) DIRTY DATASET -> dirty (dirt-clean-areas, clean dirt-clean-areas)
  if [[ -d "$DIRTY_SRC/$split" ]]; then
    for cname in "dirt-clean-areas" "clean dirt-clean-areas"; do
      if [[ -d "$DIRTY_SRC/$split/$cname" ]]; then
        echo "  + From ditry_dataset/$cname -> dirty"
        safe_cname="$(sanitize "$cname")"
        copy_images "$DIRTY_SRC/$split/$cname" "$OUT/$split/dirty" "dirtyDS_${safe_cname}"
      fi
    done
    # 3) Optional: include any 'clean' from dirty dataset into clean
    if [[ -d "$DIRTY_SRC/$split/clean" ]]; then
      echo "  + From ditry_dataset/clean -> clean"
      copy_images "$DIRTY_SRC/$split/clean" "$OUT/$split/clean" "dirtyDS_clean"
    fi
  fi
done

echo "\nDone. Sample counts:"
for split in train valid test; do
  for cls in clean dirty; do
    if [[ -d "$OUT/$split/$cls" ]]; then
      cnt=$(find "$OUT/$split/$cls" -type f | wc -l | tr -d ' ')
      echo "  $split/$cls: $cnt files"
    fi
  done
done

