# GALA

GALA policy inference and RoboCasa GR1 tabletop evaluation. This release contains environment setup, checkpoint loading, action inference, and simulation evaluation. Training pipelines, training launch scripts, datasets, and model weights are not included.

**Checkpoint release: pending.** The Hugging Face identifier below is a placeholder, not an available model. Evaluation requires a compatible GALA inference checkpoint. Once weights are published, replace `MODEL_PATH` with the released repository ID; the evaluation interface stays the same.

## Environment

Use Linux, Python 3.10, an NVIDIA GPU, and a driver compatible with CUDA 12.4. The reference parallel evaluation uses eight A800 80 GB GPUs and three inference workers per GPU. Fewer GPUs/workers are supported; memory needs depend on concurrent workers. Headless MuJoCo rendering requires working EGL/OpenGL system libraries. Building FlashAttention requires a CUDA toolkit and a C++ build toolchain.

```bash
git clone git@github.com:PuzhenYuan/GALA.git
cd GALA
conda create -n gala python=3.10 -y
conda activate gala
bash examples/environment_setup.sh
```

The setup script installs PyTorch 2.5.1 / torchvision 0.20.1 (CUDA 12.4), Transformers 4.52.0, the pinned inference requirements, FlashAttention 2.7.1.post4, and the simulator assets. It installs into the active environment; use a dedicated environment. Internet access and sufficient disk space for simulator assets and model files are required.

Simulator sources are pinned to:

| Dependency | Revision |
| --- | --- |
| robosuite | `a071383d53568ab798eb315c0e95357911be922d` |
| robocasa-gr1-tabletop-tasks | `4840e671596f93ca03651524b9f72ffb1aadfeff` |

The included RoboCasa patch matches the reference environment's exclusion of one basket asset from split B. It does not change the ID task list.

## Checkpoint

Use either a local checkpoint directory or a Hugging Face model repository:

```bash
export MODEL_PATH="./checkpoints/gala"
# After the checkpoint is released, replace the placeholder:
# export MODEL_PATH="REPLACE_WITH_HF_ORG/REPLACE_WITH_GALA_MODEL"
# Optional: pin the model revision for reproducibility.
# export GALA_MODEL_REVISION="REPLACE_WITH_COMMIT"
```

`hf://organization/model` is also supported. Remote weights are resolved once before parallel workers start. Private/gated repositories require authentication with Hugging Face. For offline use, download all checkpoint and backbone files beforehand and point to local directories.

The inference checkpoint must contain:

```text
checkpoint/
  config.json
  model.safetensors                       # or all sharded safetensors + their index
  experiment_cfg/
    metadata.json                         # GR1 modality metadata and normalization statistics
```

The configuration uses `model_type: "gala"`, `architectures: ["GALAModel"]`, and `bridge_cfg` for bridge settings. Preserve the trained action-head configuration, eight bridge tokens, image type embeddings, and GR1 normalization statistics. Weight tensor names and shapes are unchanged. Do not include optimizer states, training state, logs, datasets, or machine-specific paths in a published checkpoint.

`backbone_cfg.eagle_path` must be a Hugging Face model ID or a directory relative to the checkpoint. The reference backbone is `Qwen/Qwen2.5-VL-3B-Instruct`. Its processor/tokenizer and pretrained model files must be available because the existing architecture initializes the backbone before loading policy weights. To use a local copy:

```bash
export GALA_BACKBONE_PATH="./checkpoints/Qwen2.5-VL-3B-Instruct"
```

Inference does not need the latent encoder checkpoint or training datasets. Bridge supervision is disabled during inference. Checkpoint loading errors are fatal; the loader does not fall back to random initialization.

## Reproduce the parallel ID evaluation

Run from the repository root after installing the environment and setting `MODEL_PATH`:

```bash
mkdir -p outputs
set -o pipefail
PYTHON_BIN="$(command -v python)" \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROCS_PER_GPU=3 \
PORT_BASE=5810 \
N_ENVS=1 \
N_EPISODES=50 \
EVAL_TAG=_dexlam_all_parallel \
DATA_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only \
bash examples/run_eval_parallel.sh "$MODEL_PATH" id \
2>&1 | tee outputs/eval_id.log
```

This assigns 24 ID tasks to 24 workers: GPU `rank % 8`, ports 5810–5833, one simulator environment per worker, and 50 episodes per task (1,200 episodes total). The ID split contains six close tasks and eighteen novel-object placement tasks from split A. All task names and their order are in `examples/run_eval.sh`.

Inference defaults match the reference evaluation: absolute joint actions, a 16-step action chunk, four flow-matching denoising steps, sampling shift 1.0, and 720 maximum simulator steps per episode. The data configuration applies the reference ego-camera crop, state sine/cosine representation, and action normalization. Evaluation is stochastic; this release preserves the evaluation procedure and does not promise identical aggregate success rates on every run.

Check assignments without loading weights or starting workers:

```bash
EVAL_PARALLEL_DRY_RUN=1 GPU_IDS=0,1,2,3,4,5,6,7 PROCS_PER_GPU=3 \
bash examples/run_eval_parallel.sh "$MODEL_PATH" id
```

For a small runtime check:

```bash
GPU_IDS=0 PROCS_PER_GPU=1 EVAL_MAX_TASKS=1 N_EPISODES=1 EVAL_TAG=_smoke \
bash examples/run_eval_parallel.sh "$MODEL_PATH" id
```

Outputs are written under `outputs/evaluation_sim_id_1envs_dexlam_all_parallel/`, with one `p<rank>of24` subdirectory per worker. The directory contains per-worker videos and logs, merged client logs, task assignments, and `results.json`. Worker launcher logs are under `dev/log/`. Override `OUTPUT_ROOT` to select another output directory. Checkpoints are read-only inputs. Use a new `EVAL_TAG` for a new run; `RESUME_EVAL=1` skips completed shards, so only reuse it with the same checkpoint, settings, and worker count.

`results.json` reports per-task success rates and their average. Inspect logs for missing tasks or failed episodes before interpreting the average. Use free ports and reduce `PROCS_PER_GPU` if GPU memory is insufficient. `MUJOCO_GL=egl` is the default rendering backend.

## Standalone inference server

```bash
python scripts/inference_service.py --server \
  --model-path "$MODEL_PATH" --port 5810 \
  --data-config fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only \
  --denoising-steps 4 --infer-sample-shift 1.0
```

The ZMQ server binds to loopback by default. `gala.eval.robot.RobotInferenceClient` exposes `get_modality_config()` and `get_action(observations)`. Observations contain the ego RGB image, left/right arm joints, left/right hand joints, waist joints, and a language instruction. Query the server's modality configuration for exact keys and temporal indices. Actions are returned as per-modality joint action chunks.

## Attribution

This package retains the applicable NVIDIA GR00T copyright headers, Apache-2.0 license, and third-party notices. See `LICENSE` and `NOTICE.txt`. RoboCasa, robosuite, Qwen, and their assets or weights retain their respective upstream licenses.
