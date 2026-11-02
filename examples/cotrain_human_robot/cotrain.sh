#!/usr/bin/env bash
# Co-train GR00T N1.7 on a human dataset and a robot dataset with different action spaces.
# Each dataset has its own embodiment tag (and so its own state/action projector):
#   human -> new_embodiment   (24-d EEF pose + hand)
#   robot -> new_embodiment_2 (26-d arm joints + hand)
#
# Usage: bash examples/cotrain_human_robot/cotrain.sh [-- <extra launch_finetune.py args>...]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
HUMAN_DATASET="${HUMAN_DATASET:-data/pick_and_place2_ee_v2.1}"
ROBOT_DATASET="${ROBOT_DATASET:-data/ego_vdrd_merged_v2.1}"
HUMAN_MIX_RATIO="${HUMAN_MIX_RATIO:-0.5}"
ROBOT_MIX_RATIO="${ROBOT_MIX_RATIO:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/cotrain_human_robot3}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-100000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

# Install the per-dataset modality.json (keeping a backup of any different existing one).
install_modality() {
    local src="$1" dataset="$2" dst="$2/meta/modality.json"
    if [ -f "$dst" ] && ! cmp -s "$src" "$dst" && [ ! -f "$dst.bak" ]; then
        cp "$dst" "$dst.bak"
    fi
    cp "$src" "$dst"
}
install_modality "$HERE/modality_human.json" "$HUMAN_DATASET"
install_modality "$HERE/modality_robot.json" "$ROBOT_DATASET"

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$HUMAN_DATASET"
    --embodiment_tag new_embodiment
    --mix_ratio "$HUMAN_MIX_RATIO"
    --extra_datasets "$ROBOT_DATASET:new_embodiment_2:$ROBOT_MIX_RATIO"
    --modality_config_path "$HERE/cotrain_config.py"
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTPUT_DIR"
    --max_steps "$MAX_STEPS"
    --save_steps 1000
    --save_total_limit 5
    --warmup_ratio 0.05
    --weight_decay 1e-5
    --learning_rate 1e-4
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --dataloader_num_workers 4
    --state_dropout_prob 0.3
    # Head (600x960) and wrist (480x640) cameras differ in aspect ratio; pad to square so
    # the views of one sample can be stacked.
    --letter_box_transform
)

if [ "${1:-}" = "--" ]; then
    shift
    LAUNCH_CMD+=("$@")
fi

cd "$REPO_ROOT"
if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec uv run python "${LAUNCH_CMD[@]}"
fi
exec uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_CMD[@]}"
