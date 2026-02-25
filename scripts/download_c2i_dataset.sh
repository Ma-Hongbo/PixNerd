#!/usr/bin/env bash
set -euo pipefail

# Download a ready-to-use C2I ImageFolder dataset for quick training.
# Default dataset: Imagenette (10 classes), structure:
#   datasets/imagenette2-320/train/<class_name>/*.jpeg
#   datasets/imagenette2-320/val/<class_name>/*.jpeg
#
# Usage:
#   bash scripts/download_c2i_dataset.sh
#   bash scripts/download_c2i_dataset.sh datasets
#
# Note:
#   Full ImageNet-1K usually requires manual credentialed download.
#   This script provides an open quick-start dataset for C2I debugging/training.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,30p' "$0"
  exit 0
fi

TARGET_ROOT="${1:-datasets}"
DATASET_NAME="imagenette2-320"
URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

mkdir -p "${TARGET_ROOT}"
ARCHIVE_PATH="${TARGET_ROOT}/${DATASET_NAME}.tgz"
EXTRACTED_DIR="${TARGET_ROOT}/${DATASET_NAME}"

if [[ -d "${EXTRACTED_DIR}/train" && -d "${EXTRACTED_DIR}/val" ]]; then
  echo "[INFO] Dataset already exists: ${EXTRACTED_DIR}"
  exit 0
fi

echo "[INFO] Downloading ${DATASET_NAME} to ${ARCHIVE_PATH}"
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${ARCHIVE_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE_PATH}" "${URL}"
else
  echo "[ERROR] Neither curl nor wget is available."
  exit 1
fi

echo "[INFO] Extracting archive..."
tar -xzf "${ARCHIVE_PATH}" -C "${TARGET_ROOT}"

echo "[INFO] Dataset ready:"
echo "       Train: ${EXTRACTED_DIR}/train"
echo "       Val  : ${EXTRACTED_DIR}/val"
echo
echo "[INFO] Next step:"
echo "       bash scripts/train_c2i_wandb_100step.sh fit ${EXTRACTED_DIR}/train"
