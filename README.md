# VLA Benchmarking

A benchmarking harness for evaluating VLA (vision-language-action) policies. Ships policy
clients for GR00T, OpenPI (pi0 / pi0.5), and MolmoAct, plus the evaluation loop, planning,
and results/video I/O needed to run them against a robot.

## Installation

### Client-only (no real robot)

If you're only exercising the policy clients themselves (e.g. against a running GR00T /
OpenPI / MolmoAct server, without driving real hardware), install into an isolated
environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates a `.venv` and installs `VLA_benchmarking` plus all of its pure-Python
dependencies. `eval_control.py` (the piece that actually drives a robot) will *not* be
importable from this environment -- see the next section.

### Real-robot evaluation

Running an evaluation against a physical robot requires the
[DROID robot platform](https://github.com/droid-dataset/droid) to already be set up
according to DROID's own [setup guide](https://droid-dataset.github.io/droid) -- hardware
assembly, the NUC-side install, calibration, etc. DROID is **not** a dependency of this
package: it's a hardware-setup repo (pulls in `polymetis`, `mujoco`, camera SDKs, and other
robot-specific tooling) that was never meant to be `pip install`-able, so we don't try to
manage it here.

Once DROID's control-machine environment exists (their own conda env, typically), install
`VLA_benchmarking` into *that* environment instead of creating a separate `uv`-managed one:

```bash
conda activate <your-droid-env>
pip install -e /path/to/VLA_benchmarking
```

Because both packages now live in the same environment, `from droid.robot_env import
RobotEnv` in `eval_control.py` resolves normally -- no `PYTHONPATH` tricks needed. If DROID
isn't importable, you'll get a clear `ImportError` pointing back here rather than a bare
`ModuleNotFoundError`.

`pyproject.toml` is standard PEP 621 packaging metadata, so `pip`, `uv`, or any other
PEP 517-compliant installer works identically against it -- `uv sync` is just the fast path
for the isolated, client-only case above.

## Repository layout

- `src/VLA_benchmarking/` -- the installable package: evaluation loop (`eval_control.py`),
  planning (`eval_planning.py`), results/video I/O (`eval_io.py`), config (`system_config.py`),
  and the policy clients (`policy_clients/`).
- `configs/` -- policy, task, and eval configs (YAML).
- `scripts/run_eval.py` -- entry point for running an evaluation.

## Supported policies

Policy clients are registered by name in `AbstractPolicyClient._registry` and selected via
each policy's config YAML (`configs/policy_configs/`):

- `gr00t` -- GR00T, served over ZMQ
- `pi0` / `pi05` -- OpenPI, served over the openpi WebSocket protocol
- `molmoact` -- MolmoAct, served over HTTP

Each of these talks to a policy server running elsewhere (its own environment, typically
GPU-equipped) over the network -- the policy server's own model/training code is not a
dependency of this repo either, for the same reason DROID isn't.
