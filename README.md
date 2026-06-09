# MV-Actor

Evaluation-only release for MV-Actor on bimanual RLBench / PerAct2 tasks.

This repository contains policy evaluation code only. It does not include training code, datasets, simulator binaries, or model checkpoints.

## Installation

Create a conda environment and install the Python dependencies:

```bash
conda create -y --name mv_actor python=3.10
conda activate mv_actor
pip install torch torchvision torchaudio
pip install -r requirements-eval.txt
```

Install the PerAct2-compatible simulator stack:

```bash
git clone https://github.com/markusgrotz/PyRep.git
git clone https://github.com/markusgrotz/RLBench.git
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Follow the PyRep/RLBench install steps from the original 3DFA repository:
https://github.com/nickgkan/3d_flowmatch_actor

## Data Preparation

### PerAct2

Download the official 3DFA PerAct2 data and test seeds using:

```bash
bash scripts/rlbench/peract2_datagen_fast.sh
```

This follows the official 3DFA data script:
https://github.com/nickgkan/3d_flowmatch_actor/blob/master/scripts/rlbench/peract2_datagen_fast.sh

The script downloads:

- PerAct2 zarr data: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
- PerAct2 test seeds: https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip

Set `DATA_DIR` to the local test seed directory:

```bash
export DATA_DIR=/path/to/peract2_raw/peract2_test
```

## Checkpoints

Download the MV-Actor evaluation checkpoint from the release assets, then set:

```bash
export CKPT=/path/to/best_iter200000_snapshot.pth
```

For the original 3DFA baseline, use the official PerAct2 checkpoint:
https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth

The checkpoint loader expects:

```python
{
    "weight": state_dict,
    "ema_weight": None,
    "iter": 200000,
    "best_loss": float,
}
```

Only `ckpt["weight"]` is loaded for evaluation. The `iter` field is metadata.

## Online Evaluation

Set the local paths first:

```bash
export CKPT=/path/to/best_iter200000_snapshot.pth
export DATA_DIR=/path/to/peract2_raw/peract2_test
export CLIP_HF_LOCAL_PATH=/path/to/local/openai_clip_vit_base_patch32_snapshot
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/path/to/RLBench
export PYREP_ROOT=/path/to/PyRep
```

Run PerAct2 evaluation:

```bash
bash online_evaluation_rlbench/eval_peract2.sh
```

The full evaluation arguments are in:

```text
online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh
```

## What Is Included

- `online_evaluation_rlbench/eval_peract2.sh`: public PerAct2 evaluation entry point.
- `online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh`: full MV-Actor PerAct2 evaluation arguments.
- `online_evaluation_rlbench/evaluate_policy.py`: online RLBench evaluation code.
- `modeling/`: model definition required for checkpoint loading.
- `datasets/`: PerAct / PerAct2 camera and task metadata.
- `utils/`: geometry and tensor utilities used at evaluation time.

## What Is Not Included

- No `.pth`, `.pt`, `.ckpt`, or `.safetensors` checkpoint files.
- No RLBench / CoppeliaSim / PyRep simulator binaries.
- No datasets or evaluation logs.
- No training scripts.

## Notes

This first release supports raw point-cloud online evaluation. Pi3X/VGGT online geometry evaluation is not included.

## Citation

This code is built on top of the official 3DFA implementation:
https://github.com/nickgkan/3d_flowmatch_actor
