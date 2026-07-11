# VLA Benchmarking on DROID

A benchmarking framework for evaluating vision-language-action (VLA) policies — PI0, PI05, MolmoAct2, Gr00t — on real-world manipulation tasks using the [DROID](https://github.com/droid-dataset/droid) robot platform.

This repository contains the evaluation code accompanying the "A Few Words Go a Long Way: Language Guided Robot Policy Synthesis" paper. The paper is awaiting peer review and has not been published. More information can be found on the paper's [website](https://robo-architect.github.io/).

> **Note:** This repository is designed to run exclusively against a physical DROID robot rig. There is no support for simulation or other embodiments.

## Overview

At a high level, the evaluation loop works as follows:

- Each VLA policy (pi0, pi05, MolmoAct, GR00T) is served remotely over HTTP by its own policy server; this repo does not run policy inference in-process.
- Policy identity and connection details (checkpoint, host, port, action horizon, etc.) are defined in per-policy YAML config files.
- `run_eval.py` is the main entrypoint: it connects to the DROID robot environment, builds the appropriate policy client, and executes an evaluation plan.
- Evaluation plans and individual tasks are defined in YAML config files (see `configs/` directory).
- `compute_results.py` aggregates saved rollout results into summary scores.

## Prerequisites

Running an evaluation with this repository requires:

1. **A physical DROID robot rig**, fully assembled and calibrated. 
2. **The DROID software stack** installed and working in a Python environment. This repo does not install or vendor DROID; see the [DROID repository](https://github.com/droid-dataset/droid) for setup instructions. `import droid` must succeed before anything here will run.
3. **This repository's dependencies**, installed into that same DROID environment.
4. **A running policy server** for whichever VLA policy you want to evaluate (pi0, pi05, MolmoAct, or GR00T), reachable over HTTP from the machine running the DROID env.

## Setup

### 1. Set up the DROID software stack

Follow the official DROID software setup guide:

https://droid-dataset.github.io/droid/docs/software-setup

> **Note:** This repository has not been validated against DROID's Docker-based setup instructions. We recommend setting up on host.

### 2. Clone this repository

Clone this repository into your workspace.

```
git clone https://github.com/EricJGoossen/VLA_benchmarking.git
```

### 3. Install this repo's dependencies into the DROID conda env

Activate the conda environment you set up on the laptop/workstation for droid in step 1, then install this repository's dependencies into it:

```bash
conda activate robot
cd VLA_benchmarking
pip install .
```

### 4. Determine camera ID

Open zed explorer, which you should have setup during step 1. The camera ID should be next to its name in the viewer. Edit the respective camera ID fields inside [system_config.py](src/VLA_benchmarking/system_config.py) to match the values you just found. While there, you can also configure the default CLI arguements for any other field.

### 5. Set up policy servers

Each VLA policy runs as its own remote server, served over HTTP and accessed via this repo's policy clients. For each policy you plan to evaluate, clone and follow the setup instructions for the pinned version used in our paper:

| Policy | Repository | Pinned commit |
|---|---|---|
| pi0 / pi0.5 | [xuningy/openpi](https://github.com/xuningy/openpi) (fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) @ [`175f89c`](https://github.com/Physical-Intelligence/openpi/commit/175f89c31d1b2631a8ff3b678768f17489c5ead4)) | [`aa64205`](https://github.com/xuningy/openpi/commit/aa6420561529593114160d05e5ad155792b272f3) |
| MolmoAct | [allenai/MolmoAct2](https://github.com/allenai/MolmoAct2) | [`2e88152`](https://github.com/allenai/MolmoAct2/commit/2e88152e396b1250abf8a6ac66ce666dde1dc1ed) |
| GR00T | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | [`626af89`](https://github.com/NVIDIA/Isaac-GR00T/commit/626af89d3e914ec92eab5323e23b9ed44a7b26c8) |

Each policy server is typically set up in its **own environment**, separate from the DROID conda env to prevent dependency conflicts.



## Usage

Running an evaluation involves three separate processes, typically on at least two machines: the DROID **NUC** (robot controller), and your **DROID workstation/laptop** (where this repo and the VLA policy servers run).

### 1. Start the robot server (on the NUC)

Run this from wherever you cloned the `droid` repository on the NUC. Leave this running for the duration of your session.

```bash
conda activate polymetis-local
python3 scripts/server/run_server.py
```

### 2. Start a policy server (on your workstation)

Each VLA policy is served independently, in its own environment, separate from the `robot` conda env. Pick the policy you want to evaluate:

**pi0 / pi0.5** — from inside your `openpi` clone:

```bash
# pi0.5
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

# pi0
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid
```

**MolmoAct** — from inside your `MolmoAct2` clone:

```bash
uv run python examples/droid/host_server_droid.py --dtype bfloat16
```

**GR00T** — from inside your `Isaac-GR00T` clone:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0
```

Each command above accepts extra args to configure host/port (check each server's `--help` for the exact flags). The default host and port for each server has been configured in its respective config file, but if you change from the default, you must match the host and port as CLI arguements in step 3.

### 3. Run an evaluation (this repo)

Activate the `robot` conda env (where this repo was `pip install`ed per the Setup section), then run:

```bash
conda activate robot
python scripts/run_eval.py --policy-config configs/policy_configs/PI05.yaml --config-file configs/eval_configs/full_benchmark_eval.yaml
```

**Required argument:**

| Flag | Description |
|---|---|
| `--policy-config` | Path to the policy YAML for the server you started in step 2 (e.g. `configs/policy_configs/PI05.yaml`, `PI0.yaml`, `MolmoAct2.yaml`, `Gr00t.yaml`). |

**Optional arguments:**

| Flag | Description |
|---|---|
| `--config-file` | Path to a task config (`configs/task_configs/*.yaml`, runs one episode) or an evaluation config (`configs/eval_configs/*.yaml`, runs a full sequence of episodes). If omitted, runs an interactive test loop instead — no data or scores are saved, useful for sanity-checking a policy server connection before a real run. |
| `--server-host` / `--server-port` | Override the `default_remote_host` / `default_remote_port` from the policy config, if your policy server is running on a different host/port than the config specifies. |
| `--results-dir` | Directory to write results into. If omitted, defaults to a timestamped folder under `--default-results-dir` (default `./results`), named after the evaluation or `{policy}_{task}`. |
| `--scene-camera-id` / `--wrist-camera-id` | Override the camera IDs set in `system_config.py` without editing the file. |
| `--recording-fps` | Frame rate for saved rollout videos (default 10). |
| `--record-scene-camera` / `--record-wrist-camera` | Toggle whether each camera's video is saved (default: both `True`). |

Once running, the tool will prompt you before each rollout, then ask for a success flag, step score, recall score, and optional comments after each one. If interrupted mid-episode (Ctrl+C), you'll be asked whether to resume, advance to the next episode (evaluation configs only), run a test rollout, or quit. A partially-completed run cane be resumed later later since `run_eval.py` checks on-disk results and skips completed rollouts.

### 4. Aggregate results

Once evaluation(s) are complete for one or more policies under a shared results directory:

```bash
python scripts/compute_results.py <results_dir> <output_dir>
```

This produces, per policy, `raw.csv` (one row per rollout), `by_task.csv`, and `by_prompt.csv` (success rate and normalized scores averaged by task/instruction) under `<output_dir>/{policy}_results/`.

## Repository Structure

| File | Purpose |
|---|---|
| `scripts/run_eval.py` | Main entrypoint; builds the policy client and DROID environment, then runs an evaluation plan (or a test loop). |
| `src/VLA_benchmarking/eval_control.py` | Drives the rollout loop against the robot: stepping the policy, recording video/state, and saving per-rollout results. |
| `src/VLA_benchmarking/eval_planning.py` | Parses evaluation/task config files and result directory state into an executable plan of episodes. |
| `src/VLA_benchmarking/eval_io.py` | Shared I/O utilities: loading policy/task configs, saving videos, writing result files. |
| `src/VLA_benchmarking/eval_ui.py` | User-facing prompts and status messages: score input, rollout start/stop confirmations, test-loop instruction entry. |
| `src/VLA_benchmarking/system_config.py` | CLI argument / config dataclass definitions (`Args`) and config schema key lists. |
| `src/VLA_benchmarking/policy_clients/abstract_policy_client.py` | Base class for all policy clients; policy registry and `from_config` construction logic. |
| `src/VLA_benchmarking/policy_clients/openpi_client.py` | Client for pi0 / pi0.5, served via the openpi WebSocket server. |
| `src/VLA_benchmarking/policy_clients/molmoact_client.py` | Client for MolmoAct, served via an HTTP REST server. |
| `src/VLA_benchmarking/policy_clients/groot_client.py` | Client for GR00T, served via a ZMQ server. |
| `scripts/compute_results.py` | Aggregates saved rollout results into summary scores. |
| `configs/` | Per-policy YAML configs and per-task/evaluation YAML configs. |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.