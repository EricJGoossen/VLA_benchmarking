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

*Usage instructions for running evaluations via `run_eval.py`, including policy config and task config files — coming soon.*

## Repository Structure

| File | Purpose |
|---|---|
| `run_eval.py` | Main entrypoint; builds the policy client and DROID environment, then runs an evaluation plan (or a test loop). |
| `eval_control.py` | Drives the rollout loop against the robot: stepping the policy, recording video/state, and saving per-rollout results. |
| `eval_planning.py` | Parses evaluation/task config files and result directory state into an executable plan of episodes. |
| `eval_io.py` | Shared I/O utilities: loading policy/task configs, saving videos, writing result files. |
| `policy_clients.py` | `PolicyClient` implementations for each supported VLA policy (pi0, pi05, MolmoAct, GR00T), communicating with remote policy servers. |
| `system_config.py` | CLI argument / config dataclass definitions (`Args`). |
| `compute_results.py` | Aggregates saved rollout results into summary scores. |
| `configs/` | Per-policy YAML configs and per-task/evaluation YAML configs. |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.