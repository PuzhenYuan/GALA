# GALA: Geometry-Aware Latent Action Modeling for Vision-Language-Action Model Pretraining across Embodiments

[![GALA teaser](figure/Teaser.png)](figure/Teaser.pdf)

Official repository for GALA: Geometry-Aware Latent Action Modeling for Vision-Language-Action Model Pretraining across Embodiments

[![Project Page](https://img.shields.io/badge/Project-Page-Blue)](https://puzhenyuan.github.io/GALA-website/) [![arXiv](https://img.shields.io/badge/arXiv-2609.21948-b31b1b)](https://arxiv.org/abs/2609.21948v1) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoint-yellow)](https://huggingface.co/ypz21/GALA_robocasa_gr1)

## Setup

Use Linux, Python 3.10, an NVIDIA GPU with CUDA 12.4 support, a CUDA build toolchain, and EGL for headless rendering.

```bash
git clone git@github.com:PuzhenYuan/GALA.git
cd GALA
conda create -n gala python=3.10 -y
conda activate gala
bash examples/environment_setup.sh
```

## Download checkpoints

```bash
hf auth login  # Use an account with access to this private model repository.
hf download ypz21/GALA_robocasa_gr1 --local-dir checkpoints/checkpoint_robocasa_gr1
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir checkpoints/Qwen2.5-VL-3B-Instruct
```

## Evaluate

Run from the GALA repository root:

```bash
mkdir -p outputs
set -o pipefail
PYTHON_BIN="$(command -v python)" \
GALA_BACKBONE_PATH="$PWD/checkpoints/Qwen2.5-VL-3B-Instruct" \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROCS_PER_GPU=3 \
PORT_BASE=5810 \
N_ENVS=1 \
N_EPISODES=50 \
EVAL_TAG=_gala_parallel \
DATA_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only \
bash examples/run_eval_parallel.sh checkpoints/checkpoint_robocasa_gr1 id \
2>&1 | tee outputs/eval_gala_robocasa_gr1_id.log
```

This runs 24 ID tasks with 50 episodes each, using 8 GPUs and 3 workers per GPU. Results and videos are saved under `outputs/evaluation_sim_id_1envs_gala_parallel/`.

For a quick check, use `GPU_IDS=0`, `PROCS_PER_GPU=1`, `N_EPISODES=1`, `EVAL_MAX_TASKS=1`, and `EVAL_TAG=_smoke`. The checkpoint argument also accepts `ypz21/GALA_robocasa_gr1` directly. No training data or latent-encoder weights are required.

## Citation

```bibtex
@misc{liu2026galageometryawarelatentaction,
  title={GALA: Geometry-Aware Latent Action Modeling for Vision-Language-Action Model Pretraining across Embodiments},
  author={Yichen Liu and Puzhen Yuan and Xiang Zhu and Yanjiang Guo and Jianyu Chen},
  year={2026},
  eprint={2609.21948},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.21948},
}
```
