# Reproducibility Checklist

## 1. Installation

```bash
git clone https://github.com/TianYinchen56/MV-Actor.git
cd MV-Actor
conda create -y --name mv_actor python=3.10
conda activate mv_actor
pip install torch torchvision torchaudio
pip install -r requirements-eval.txt
```

Install PyRep, RLBench, and CoppeliaSim:

```text
PyRep:       https://github.com/markusgrotz/PyRep
RLBench:     https://github.com/markusgrotz/RLBench
CoppeliaSim: https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

## 2. Checkpoints and Weights

```bash
bash scripts/weights/download_mv_actor_checkpoint.sh
bash scripts/weights/download_clip_weights.sh
bash scripts/weights/download_pi3x_weights.sh
```

If the GitHub repository is private, set `GITHUB_TOKEN` before running the checkpoint script.

MV-Actor checkpoint:

```text
https://github.com/TianYinchen56/MV-Actor/releases/download/v0.1-eval/best_iter200000_snapshot.pth
```

SHA256:

```text
2b5655e217b20fada49ba6757e4158e276b04a504ad8d9bbaf8b28e5888f6793
```

## 3. Code

```text
scripts/rlbench/peract2_datagen_fast.sh
scripts/weights/download_mv_actor_checkpoint.sh
scripts/weights/download_clip_weights.sh
scripts/weights/download_pi3x_weights.sh
online_evaluation_rlbench/eval_peract2.sh
online_evaluation_rlbench/run_eval_mv_actor_alltasks.sh
online_evaluation_rlbench/evaluate_policy.py
online_evaluation_rlbench/evaluate_policy_inprocess_pi3x.py
online_evaluation_rlbench/pi3x_worker.py
```

## 4. Commands

Download PerAct2 data:

```bash
bash scripts/rlbench/peract2_datagen_fast.sh
```

Set paths:

```bash
export CKPT=$PWD/checkpoints/best_iter200000_snapshot.pth
export DATA_DIR=$PWD/peract2_raw/peract2_test
export CLIP_RN50_LOCAL_PATH=$PWD/third_party_weights/openai_clip/RN50.pt
export CLIP_HF_LOCAL_PATH=$PWD/third_party_weights/openai_clip_vit_base_patch32
export PI3_ROOT=$PWD/third_party/Pi3
export PI3X_CKPT=$PWD/third_party_weights/pi3x/model.safetensors
export COPPELIASIM_ROOT=/absolute/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/absolute/path/to/RLBench
export PYREP_ROOT=/absolute/path/to/PyRep
```

Run evaluation:

```bash
bash online_evaluation_rlbench/eval_peract2.sh
```
