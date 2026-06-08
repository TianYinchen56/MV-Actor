#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:?Set CKPT to a local MV-Actor checkpoint path.}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the local PerAct2 raw test dataset path.}
CLIP_HF_LOCAL_PATH=${CLIP_HF_LOCAL_PATH:?Set CLIP_HF_LOCAL_PATH to a local CLIP Transformers snapshot.}
COPPELIASIM_ROOT=${COPPELIASIM_ROOT:?Set COPPELIASIM_ROOT.}
RLBENCH_ROOT=${RLBENCH_ROOT:?Set RLBENCH_ROOT.}
PYREP_ROOT=${PYREP_ROOT:?Set PYREP_ROOT.}

PY=${PY:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/mv_actor_eval_$(date +%Y%m%d_%H%M%S)}
DATASET=${DATASET:-Peract2_3dfront_3dwrist}
IMAGE_SIZE=${IMAGE_SIZE:-256,256}
FPS_SUBSAMPLING_FACTOR=${FPS_SUBSAMPLING_FACTOR:-5}
DISPLAY_NUM=${DISPLAY_NUM:-300}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CKPT DATA_DIR CLIP_HF_LOCAL_PATH COPPELIASIM_ROOT RLBENCH_ROOT PYREP_ROOT CUDA_VISIBLE_DEVICES
export PYTHONPATH="$RLBENCH_ROOT:$PYREP_ROOT:$(pwd):${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASKS=(
  bimanual_push_box
  bimanual_lift_ball
  bimanual_dual_push_buttons
  bimanual_pick_plate
  bimanual_put_item_in_drawer
  bimanual_put_bottle_in_fridge
  bimanual_handover_item
  bimanual_pick_laptop
  bimanual_straighten_rope
  bimanual_sweep_to_dustpan
  bimanual_lift_tray
  bimanual_handover_item_easy
  bimanual_take_tray_out_of_oven
)

get_var_arg() {
  case "$1" in
    bimanual_dual_push_buttons|bimanual_put_item_in_drawer|bimanual_handover_item) echo -1 ;;
    *) echo 0 ;;
  esac
}

get_demo_arg() {
  case "$1" in
    bimanual_dual_push_buttons|bimanual_put_item_in_drawer|bimanual_handover_item) echo -1 ;;
    *) echo 100 ;;
  esac
}

mkdir -p "$OUTPUT_ROOT"
for task_name in "${TASKS[@]}"; do
  task_dir="$OUTPUT_ROOT/$task_name"
  mkdir -p "$task_dir"
  variation_arg=$(get_var_arg "$task_name")
  demo_arg=$(get_demo_arg "$task_name")
  echo "[start] task=$task_name variation=$variation_arg demo_amount=$demo_arg output=$task_dir"
  DISPLAY_NUM="$DISPLAY_NUM" XVFB_LOG_FILE="$task_dir/xvfb.log" \
    bash online_evaluation_rlbench/with_xvfb.sh "$PY" online_evaluation_rlbench/evaluate_policy.py \
      --checkpoint "$CKPT" \
      --task "$task_name" \
      --variation "$variation_arg" \
      --demo_amount "$demo_arg" \
      --from_episode_number 0 \
      --max_tries 2 \
      --max_steps 25 \
      --headless true \
      --collision_checking false \
      --seed 0 \
      --data_dir "$DATA_DIR" \
      --dataset "$DATASET" \
      --image_size "$IMAGE_SIZE" \
      --output_file "$task_dir/eval.json" \
      --model_type denoise3d \
      --bimanual true \
      --prediction_len 1 \
      --backbone clip \
      --finetune_backbone false \
      --finetune_text_encoder false \
      --fps_subsampling_factor "$FPS_SUBSAMPLING_FACTOR" \
      --enable_interleaved_point_fusion true \
      --interleaved_before_language false \
      --interleaved_text_between_self_and_geo false \
      --interleaved_cross_overlap_only false \
      --interleaved_point_fusion_layers 1 \
      --interleaved_semantic_mode cross_view_reproject \
      --interleaved_cross_knn 16 \
      --interleaved_cross_residual_scale 0.1 \
      --interleaved_knn_chunk_size 256 \
      --interleaved_cross_view_radius 1 \
      --interleaved_cross_view_max_reproj_error 1.5 \
      --interleaved_cross_view_max_depth_error 0.05 \
      --pi3x_geometry_mode pointwise_sig32 \
      --geo_upsample_mode nearest_copy \
      --pointwise_parallel_branch_fusion true \
      --pointwise_parallel_branch_text_last false \
      --interleaved_semantic_use_pos false \
      --disable_semantic_fusion true \
      --disable_geometry_fusion false \
      --lv2_batch_size 1 \
      --scene_geometry_source raw_pcd \
      --online_semantic_corr_source raw_pcd \
      --embedding_dim 120 \
      --num_attn_heads 8 \
      --num_vis_instr_attn_layers 3 \
      --num_history 3 \
      --num_shared_attn_layers 4 \
      --relative_action false \
      --rotation_format quat_xyzw \
      --denoise_timesteps 5 \
      --denoise_model rectified_flow \
      > "$task_dir/task.log" 2>&1
  echo "[done] task=$task_name"
done

echo "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/output_dir.txt"
