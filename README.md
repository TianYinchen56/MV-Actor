# MV-Actor

Evaluation-only release for MV-Actor on bimanual RLBench / PerAct2 tasks.

This repository contains code for policy evaluation only. It does not include training code, datasets, simulator binaries, or model checkpoints.

## What is included

- `online_evaluation_rlbench/evaluate_policy.py`: online RLBench evaluation entry point.
- `online_evaluation_rlbench/with_xvfb.sh`: helper for headless CoppeliaSim rendering.
- `online_evaluation_rlbench/utils_with_bimanual_rlbench.py`: bimanual Dual Panda RLBench wrapper.
- `modeling/`: policy/model definition required for checkpoint loading.
- `datasets/`: PerAct / PerAct2 camera and task metadata.
- `utils/`: geometry and tensor utilities used at evaluation time.

## What is not included

- No `.pth`, `.pt`, `.ckpt`, or `.safetensors` checkpoint files.
- No RLBench / CoppeliaSim / PyRep simulator binaries.
- No datasets or evaluation logs.
- No training scripts.

## Required local assets

Set these paths before running evaluation:

```bash
export CKPT=/path/to/best_iter200000_snapshot.pth
export DATA_DIR=/path/to/peract2_raw/peract2_test
export CLIP_HF_LOCAL_PATH=/path/to/local/openai_clip_vit_base_patch32_snapshot
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/path/to/RLBench
export PYREP_ROOT=/path/to/PyRep
```

The checkpoint format expected by the loader is:

```python
{
    "weight": state_dict,
    "ema_weight": None,      # optional
    "iter": 200000,
    "best_loss": float,      # optional
}
```

The code loads only `ckpt["weight"]` for evaluation. The `iter` field is metadata and does not change model weights.

## Environment

Install the Python dependencies used by the original 3DFA/RLBench environment. A typical environment needs:

- Python 3.10 or 3.11
- PyTorch
- torchvision
- transformers
- diffusers
- einops
- numpy
- scipy
- pillow
- psutil
- threadpoolctl
- zarr
- RLBench, PyRep, and CoppeliaSim
- OpenAI CLIP Python package for the CLIP visual backbone

Make sure simulator paths are visible:

```bash
export PYTHONPATH="$RLBENCH_ROOT:$PYREP_ROOT:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

## Run one task

```bash
python online_evaluation_rlbench/evaluate_policy.py \
  --checkpoint "$CKPT" \
  --task bimanual_pick_plate \
  --variation 0 \
  --demo_amount 100 \
  --from_episode_number 0 \
  --max_tries 2 \
  --max_steps 25 \
  --headless true \
  --collision_checking false \
  --seed 0 \
  --data_dir "$DATA_DIR" \
  --dataset Peract2_3dfront_3dwrist \
  --image_size 256,256 \
  --output_file outputs/bimanual_pick_plate/eval.json \
  --model_type denoise3d \
  --bimanual true \
  --prediction_len 1 \
  --backbone clip \
  --finetune_backbone false \
  --finetune_text_encoder false \
  --fps_subsampling_factor 5 \
  --enable_interleaved_point_fusion true \
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
  --denoise_model rectified_flow
```

## Run all PerAct2 tasks

Use the provided launcher:

```bash
bash online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh
```

The launcher reads `CKPT`, `DATA_DIR`, `CLIP_HF_LOCAL_PATH`, `COPPELIASIM_ROOT`, `RLBENCH_ROOT`, and `PYREP_ROOT` from the environment.

## Notes

This release supports raw point-cloud online evaluation. Pi3X/VGGT online geometry code is intentionally not included in this first evaluation-only release.
