#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_c2i_with_viz.sh fit
#   bash scripts/train_c2i_with_viz.sh fit configs_c2i/pix256std1_repa_pixnerd_xl.yaml
#   bash scripts/train_c2i_with_viz.sh predict configs_c2i/pix256std1_repa_pixnerd_xl.yaml --ckpt_path /path/to/model.ckpt
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_c2i_with_viz.sh fit

MODE="${1:-fit}"
CONFIG="${2:-configs_c2i/pix256std1_repa_pixnerd_xl.yaml}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

has_tags_exp_override=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--tags.exp" || "${arg}" == --tags.exp=* ]]; then
    has_tags_exp_override=true
    break
  fi
done

if [[ "${MODE}" != "fit" && "${MODE}" != "predict" ]]; then
  echo "[ERROR] MODE must be fit or predict, got: ${MODE}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}"
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[INFO] Branch: ${BRANCH}"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Config: ${CONFIG}"
TAG_EXP_AUTO="$(date +c2i-%y%m%d-%H%M%S)"
echo "[INFO] tags.exp: ${TAG_EXP_AUTO}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-(none)}"

cmd=(python main.py "${MODE}" -c "${CONFIG}")
if [[ "${has_tags_exp_override}" == "false" ]]; then
  cmd+=(--tags.exp "${TAG_EXP_AUTO}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running: ${cmd[*]}"
"${cmd[@]}"
