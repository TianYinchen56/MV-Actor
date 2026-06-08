# MV-Actor Evaluation Reproducibility

This repository contains evaluation code only. It does not ship model weights, datasets, simulator binaries, or CLIP weights.

## 1. Official References

- MV-Actor evaluation code in this repository: https://github.com/TianYinchen56/MV-Actor
- Original 3DFA code: https://github.com/nickgkan/3d_flowmatch_actor
- Original 3DFA paper: https://arxiv.org/abs/2508.11002
- Official 3DFA model/data hub: https://huggingface.co/katefgroup/3d_flowmatch_actor
- Official 3DFA PerAct2 checkpoint: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth
- Official PerAct2 fast data script: https://github.com/nickgkan/3d_flowmatch_actor/blob/master/scripts/rlbench/peract2_datagen_fast.sh
- Official 3DFA PerAct2 evaluation script: https://github.com/nickgkan/3d_flowmatch_actor/blob/master/online_evaluation_rlbench/eval_peract2.sh

## 2. Required Files

Prepare these files/directories locally:

```bash
export CKPT=/absolute/path/to/best_iter200000_snapshot.pth
export DATA_DIR=/absolute/path/to/peract2_raw/peract2_test
export CLIP_HF_LOCAL_PATH=/absolute/path/to/openai_clip_vit_base_patch32_snapshot
export COPPELIASIM_ROOT=/absolute/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/absolute/path/to/RLBench
export PYREP_ROOT=/absolute/path/to/PyRep
```

`CKPT` is the MV-Actor checkpoint released by the MV-Actor authors. The checkpoint is not included in this Git repository.

For the original 3DFA baseline checkpoint, use the official PerAct2 checkpoint:

```text
https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth
```

`CLIP_HF_LOCAL_PATH` must point to a local Transformers-format CLIP snapshot containing `config.json`. The evaluator uses `local_files_only=True` and will fail if the local snapshot is missing.

## 3. Official PerAct2 Data

The original 3DFA repository provides the fast PerAct2 data script:

```bash
git clone https://github.com/nickgkan/3d_flowmatch_actor.git
cd 3d_flowmatch_actor
bash scripts/rlbench/peract2_datagen_fast.sh
```

That script downloads:

- PerAct2 training zarr: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
- PerAct2 test seeds: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip

The original script layout is:

```text
zarr_datasets/peract2/peract2/
peract2_raw/peract2_test/
```

For online evaluation in this repository, set:

```bash
export DATA_DIR=/absolute/path/to/peract2_raw/peract2_test
```

## 4. Simulator Environment

Install the PerAct2-compatible simulator stack:

- PyRep fork: https://github.com/markusgrotz/PyRep
- RLBench fork: https://github.com/markusgrotz/RLBench
- CoppeliaSim 4.1.0: https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

Expose the simulator paths before running evaluation:

```bash
export PYTHONPATH="$RLBENCH_ROOT:$PYREP_ROOT:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

## 5. Run Evaluation

Run one task:

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

Run all PerAct2 tasks serially:

```bash
bash online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh
```

## 6. What This Release Does Not Include

- Training scripts are not included.
- MV-Actor checkpoint files are not included.
- PerAct2 data is not included.
- RLBench, PyRep, and CoppeliaSim are not included.
- Pi3X/VGGT online geometry evaluation is not included in this first evaluation-only release.
