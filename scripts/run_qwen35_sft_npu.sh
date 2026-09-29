#!/usr/bin/env bash
# Run on a Linux Ascend training node with ms-swift installed.
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_qwen35_sft_npu.sh [smoke|full]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/processed/sft_clean}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/qwen35_9b_sft}"
TRAIN_FILE="$DATA_DIR/train.openai.jsonl"
DEV_FILE="$DATA_DIR/dev.openai.jsonl"

for input_file in "$TRAIN_FILE" "$DEV_FILE"; do
  if [[ ! -s "$input_file" ]]; then
    echo "Missing or empty SFT file: $input_file" >&2
    exit 1
  fi
done
if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift is not installed in this environment (swift command missing)." >&2
  exit 1
fi

# Cold-start SFT can start on 8 of the available NPUs.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
VISIBLE_COUNT="$(awk -F, '{print NF}' <<< "$ASCEND_RT_VISIBLE_DEVICES")"
if [[ "$VISIBLE_COUNT" -ne "$NPROC_PER_NODE" ]]; then
  echo "NPROC_PER_NODE=$NPROC_PER_NODE but $VISIBLE_COUNT NPU IDs are visible." >&2
  exit 1
fi

COMMON_ARGS=(
  --model "$MODEL_PATH"
  --dataset "$TRAIN_FILE"
  --val_dataset "$DEV_FILE"
  --split_dataset_ratio 0
  --loss_scale last_round
  --add_non_thinking_prefix true
  --torch_dtype bfloat16
  --tuner_type lora
  --target_modules all-linear
  --lora_rank 8
  --lora_alpha 32
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps 2
  --learning_rate 5e-5
  --max_length 2048
  --truncation_strategy delete
  --num_train_epochs 1
  --dataset_num_proc 4
  --dataloader_num_workers 4
  --logging_steps 5
  --save_total_limit 2
)

if [[ "$MODE" == "smoke" ]]; then
  echo "Running 20-step NPU smoke test on $NPROC_PER_NODE devices"
  swift sft "${COMMON_ARGS[@]}" \
    --max_steps 20 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy no \
    --output_dir "$OUTPUT_DIR/smoke"
else
  echo "Running one-epoch NPU SFT on $NPROC_PER_NODE devices"
  swift sft "${COMMON_ARGS[@]}" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --output_dir "$OUTPUT_DIR/full"
fi
