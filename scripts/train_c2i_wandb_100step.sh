#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export WANDB_API_KEY=...
#   bash scripts/train_c2i_wandb_100step.sh fit
#   bash scripts/train_c2i_wandb_100step.sh fit datasets/imagenette2-320/train my-run pixnerd-c2i
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_c2i_wandb_100step.sh fit
#
# Args:
#   $1 MODE        : fit | predict (default: fit)
#   $2 DATA_ROOT   : train imagefolder root (default: datasets/imagenette2-320/train)
#   $3 RUN_NAME    : wandb run name (default: c2i-<timestamp>)
#   $4 PROJECT     : wandb project (default: pixnerd-c2i)
#   $5 CONFIG      : config path (default: configs_c2i/pix256_c2i_wandb_100step.yaml)
#   $6 DINO_PATH   : local torch.hub DINOv2 path (default: $DINO_WEIGHT_PATH)
#   $7... EXTRA    : extra LightningCLI overrides

MODE="${1:-fit}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  sed -n '1,50p' "$0"
  exit 0
fi
DATA_ROOT="${2:-datasets/imagenette2-320/train}"
RUN_NAME="${3:-c2i-$(date +%y%m%d-%H%M%S)}"
PROJECT="${4:-pixnerd-c2i}"
CONFIG="${5:-configs_c2i/pix256_c2i_wandb_100step.yaml}"
DINO_PATH="${6:-${DINO_WEIGHT_PATH:-}}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

if [[ "${MODE}" != "fit" && "${MODE}" != "predict" ]]; then
  echo "[ERROR] MODE must be fit or predict, got: ${MODE}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[WARN] WANDB_API_KEY is empty. Wandb may run in offline/disabled mode."
fi

if [[ "${MODE}" == "fit" && ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  echo "        Run: bash scripts/download_c2i_dataset.sh"
  exit 1
fi

if [[ -z "${DINO_PATH}" ]]; then
  echo "[ERROR] DINO_PATH is empty."
  echo "        Pass arg #6 or set env DINO_WEIGHT_PATH."
  echo "        Example:"
  echo "        export DINO_WEIGHT_PATH=/path/to/facebookresearch_dinov2_main/dinov2_vitb14"
  exit 1
fi

if [[ ! -e "${DINO_PATH}" ]]; then
  echo "[ERROR] DINO_PATH not found: ${DINO_PATH}"
  exit 1
fi

echo "[INFO] Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] DINO path: ${DINO_PATH}"
echo "[INFO] Wandb project/run: ${PROJECT}/${RUN_NAME}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

cmd=(
  python main.py "${MODE}" -c "${CONFIG}"
  --data.train_dataset.init_args.root "${DATA_ROOT}"
  --model.diffusion_trainer.init_args.encoder.init_args.weight_path "${DINO_PATH}"
  --trainer.logger.init_args.project "${PROJECT}"
  --trainer.logger.init_args.name "${RUN_NAME}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
