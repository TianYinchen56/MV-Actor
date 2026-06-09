# MV-Actor Evaluation Reproducibility

This repository contains evaluation code only. It does not ship model weights, datasets, simulator binaries, or CLIP weights.

## Official References

- MV-Actor evaluation code: https://github.com/TianYinchen56/MV-Actor
- Original 3DFA code: https://github.com/nickgkan/3d_flowmatch_actor
- Original 3DFA paper: https://arxiv.org/abs/2508.11002
- Official 3DFA model/data hub: https://huggingface.co/katefgroup/3d_flowmatch_actor
- Official 3DFA PerAct2 checkpoint: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth
- Official PerAct2 data script: https://github.com/nickgkan/3d_flowmatch_actor/blob/master/scripts/rlbench/peract2_datagen_fast.sh
- Official 3DFA PerAct2 evaluation script: https://github.com/nickgkan/3d_flowmatch_actor/blob/master/online_evaluation_rlbench/eval_peract2.sh

## Required Local Assets

Set these paths before evaluation:

```bash
export CKPT=/absolute/path/to/best_iter200000_snapshot.pth
export DATA_DIR=/absolute/path/to/peract2_raw/peract2_test
export CLIP_HF_LOCAL_PATH=/absolute/path/to/openai_clip_vit_base_patch32_snapshot
export COPPELIASIM_ROOT=/absolute/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/absolute/path/to/RLBench
export PYREP_ROOT=/absolute/path/to/PyRep
```

`CKPT` is the MV-Actor checkpoint released by the MV-Actor authors. The checkpoint is not included in this Git repository.

`CLIP_HF_LOCAL_PATH` must point to a local Transformers-format CLIP snapshot containing `config.json`. The evaluator uses `local_files_only=True` and fails if the local snapshot is missing.

## PerAct2 Data

Download the official 3DFA PerAct2 data and test seeds:

```bash
bash scripts/rlbench/peract2_datagen_fast.sh
```

The script downloads:

- PerAct2 zarr data: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
- PerAct2 test seeds: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip

For online evaluation, set:

```bash
export DATA_DIR=/absolute/path/to/peract2_raw/peract2_test
```

## Simulator Environment

Install the PerAct2-compatible simulator stack:

- PyRep fork: https://github.com/markusgrotz/PyRep
- RLBench fork: https://github.com/markusgrotz/RLBench
- CoppeliaSim 4.1.0: https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

Expose the simulator paths:

```bash
export PYTHONPATH="$RLBENCH_ROOT:$PYREP_ROOT:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

## Evaluation

Run PerAct2 evaluation:

```bash
bash online_evaluation_rlbench/eval_peract2.sh
```

The full argument list is kept in:

```text
online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh
```

## Scope

- Training scripts are not included.
- MV-Actor checkpoint files are not included.
- PerAct2 data is not included.
- RLBench, PyRep, and CoppeliaSim are not included.
- Pi3X/VGGT online geometry evaluation is not included in this first evaluation-only release.
