#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Unified evaluator for SDS / VSD / SDI  (Task 1 / Task 2 / Task 3)
# Default guidance: 25 for SDS, 7.5 for VSD / SDI (override with --guidance)
# VSD / SDI options use the same names as in main.py
# ---------------------------

LOSS=""
GUIDANCE=""
STEPS=500
DEVICE=0
LR=1e-2
NEGATIVE_PROMPT="low quality"
PROMPTS_FILE=""
SAVE_ROOT="./outputs"

# ---- VSD: LoRA hyperparams ----
LORA_LR=1e-4
LORA_LOSS_WEIGHT=1.0
LORA_RANK=4

# ---- SDI: inversion hyperparams ----
INVERSION_N_STEPS=10
INVERSION_GUIDANCE_SCALE=-7.5
INVERSION_ETA=0.3
SDI_UPDATE_INTERVAL=25

usage() {
  cat <<EOF
Usage:
  ./eval.sh (--sds | --vsd | --sdi)
            [--guidance <num>] [--steps <int>] [--device <int>] [--lr <num>] [--prompts-file <path>]
            [--lora-lr <num>] [--lora-loss-weight <num>] [--lora-rank <int>]
            [--inversion-n-steps <int>] [--inversion-guidance-scale <num>]
            [--inversion-eta <num>] [--sdi-update-interval <int>]

Examples:
  ./eval.sh --sds --lr 1e-2
  ./eval.sh --vsd --guidance 7.5 --lr 5e-3 --lora-lr 1e-4 --lora-loss-weight 1.0
  ./eval.sh --sdi --guidance 7.5 --lr 0.005 --steps 1000 \
            --inversion-n-steps 10 --inversion-guidance-scale -7.5 \
            --inversion-eta 0.3 --sdi-update-interval 25
EOF
}

# ---------------------------
# Parse args
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sds) LOSS="sds"; shift ;;
    --vsd) LOSS="vsd"; shift ;;
    --sdi) LOSS="sdi"; shift ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
    # VSD
    --lora-lr) LORA_LR="$2"; shift 2 ;;
    --lora-loss-weight) LORA_LOSS_WEIGHT="$2"; shift 2 ;;
    --lora-rank) LORA_RANK="$2"; shift 2 ;;
    # SDI
    --inversion-n-steps) INVERSION_N_STEPS="$2"; shift 2 ;;
    --inversion-guidance-scale) INVERSION_GUIDANCE_SCALE="$2"; shift 2 ;;
    --inversion-eta) INVERSION_ETA="$2"; shift 2 ;;
    --sdi-update-interval) SDI_UPDATE_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$LOSS" ]]; then
  echo "[Error] You must specify one of --sds | --vsd | --sdi"
  usage
  exit 1
fi

if [[ -z "$GUIDANCE" ]]; then
  if [[ "$LOSS" == "sds" ]]; then GUIDANCE=25; else GUIDANCE=7.5; fi
fi

# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="$SAVE_ROOT/$LOSS"
mkdir -p "$SAVE_DIR"

# Remove images from previous runs so that they are not counted in the CLIP score
find "$SAVE_DIR" -maxdepth 1 -type f -name '*.png' -delete

# ---------------------------
# Prompts
# ---------------------------
DEFAULT_PROMPTS=(
  "A red bus driving on a desert road"
  "a boat in a river"
  "A cabin surrounded by forests"
  "A church beside a lake"
  "A villa close to the pool"
  "A castle next to a river"
  "A burger on the table"
  "A dog sitting on grass"
  "a cat sitting on a table"
  "A car on the road"
)
if [[ -n "$PROMPTS_FILE" ]]; then
  mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPTS_FILE")
else
  PROMPTS=("${DEFAULT_PROMPTS[@]}")
fi

echo "[*] Running $LOSS | guidance=$GUIDANCE | steps=$STEPS | lr=$LR | device=$DEVICE"
for prompt in "${PROMPTS[@]}"; do
  echo "=== Generating: $prompt ==="
  if [[ "$LOSS" == "vsd" ]]; then
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type "$LOSS" \
      --guidance_scale "$GUIDANCE" \
      --steps "$STEPS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR" \
      --lora_lr "$LORA_LR" \
      --lora_loss_weight "$LORA_LOSS_WEIGHT" \
      --lora_rank "$LORA_RANK"
  elif [[ "$LOSS" == "sdi" ]]; then
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type "$LOSS" \
      --guidance_scale "$GUIDANCE" \
      --steps "$STEPS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR" \
      --inversion_n_steps "$INVERSION_N_STEPS" \
      --inversion_guidance_scale "$INVERSION_GUIDANCE_SCALE" \
      --inversion_eta "$INVERSION_ETA" \
      --sdi_update_interval "$SDI_UPDATE_INTERVAL"
  else
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type "$LOSS" \
      --guidance_scale "$GUIDANCE" \
      --steps "$STEPS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR"
  fi
  echo "---"
done

echo "[*] Cleaning intermediate step images in $SAVE_DIR"
find "$SAVE_DIR" -maxdepth 1 -type f -name '[0-9]*.png' -delete || true

echo "[*] Running CLIP evaluation..."
python eval.py --fdir1 "$SAVE_DIR" --device "$DEVICE"
echo "[*] Done. Results at: $SAVE_DIR/eval.json"
