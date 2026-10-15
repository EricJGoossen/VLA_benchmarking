import contextlib
import datetime
import json
import os
import shutil
import signal
import time
import uuid
import h5py
import numpy as np

try:
    from droid.controllers.oculus_controller import VRPolicy
    from droid.misc.time import time_ms
    from droid.robot_env import RobotEnv
    from droid.trajectory_utils.misc import collect_trajectory
    from droid.trajectory_utils.trajectory_writer import TrajectoryWriter
except ImportError as e:
    raise ImportError(
        "Could not import 'droid'. Collecting teleop demos requires the DROID robot "
        "platform (https://github.com/droid-dataset/droid), including its oculus_reader "
        "submodule, to already be set up and installed into this Python environment."
    ) from e

from .eval_control import EvalControl
from .eval_io import write_eval_results, write_rollout_results, RolloutStatus
from .eval_planning import EpisodePlanEntry, EvaluationPlan
from .eval_ui import update_status, start_demo, confirm_keep_demo
from .system_config import Args, DEMO_RESULTS_FILENAME, DEMO_EXPECTED_FILES


TELEOP_POLICY_NAME = "teleop"


class DemoControl(EvalControl):
    """Collects teleoperated demonstrations in DROID's native format.

    Reuses the evaluation plan / resume / retake structure, but each slot is a
    demo recorded with the Oculus controller instead of a policy rollout.
    Each demo is saved to {episode_dir}/demo_{n}/ as DROID's trajectory.h5 plus
    SVO recordings, along with a metadata_<id>.json holding the instruction.

    A demo is recorded into demo_{n}_recording/ and only moved to demo_{n}/ once
    kept, so an earlier good take is never lost to a discarded retake. If
    recording is cut short by a crash (or Ctrl+C without a keep/discard answer),
    the finalized files are preserved in demo_{n}_partial_<time>/ with the
    trajectory renamed to trajectory_partial.h5, so DROID/openpi converters that
    glob for trajectory.h5 skip it until it's renamed back.
    """

    def __init__(self, args: Args, env: RobotEnv, vr_controller: VRPolicy):
        super().__init__(args, env, policy_client=None)
        self.vr_controller = vr_controller

    def record_demo(self, instruction: str, demo_num: int, num_demos: int, episode_dir: str, task_name: str, filename: str) -> None:
        """Record demo slot demo_num, repeating until a demo is kept.

        Raises KeyboardInterrupt on Ctrl+C, with a demo_saved attribute saying
        whether the operator chose to keep the interrupted demo.
        """
        demo_name = f"demo_{demo_num}"
        recording_dir = os.path.join(episode_dir, f"{demo_name}_recording")
        save_filepath = os.path.join(recording_dir, "trajectory.h5")
        recording_folderpath = os.path.join(recording_dir, "recordings")

        while True:
            start_demo(instruction, demo_num, num_demos)

            # Left over only if a previous run was killed mid-demo
            self._preserve_partial(recording_dir, episode_dir, demo_name)
            os.makedirs(recording_folderpath)

            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            metadata = {
                "task_name": task_name,
                "language_instruction": instruction,
                "demo_number": demo_num,
                "time": timestamp,
            }

            update_status(f"RECORDING demo {demo_num}... press A to SAVE, B to DISCARD.")
            start_time = time.time()
            try:
                controller_info = self._record_trajectory(save_filepath, recording_folderpath, metadata)
            except KeyboardInterrupt as e:
                duration = time.time() - start_time
                update_status("Recording stopped with Ctrl+C. (Next time, press A on the controller to save a demo.)")
                try:
                    keep = confirm_keep_demo(demo_num)
                except KeyboardInterrupt:
                    keep = None
                if keep:
                    self._keep_demo(recording_dir, episode_dir, demo_name, demo_num, instruction, metadata, duration, timestamp, filename)
                elif keep is None:
                    self._preserve_partial(recording_dir, episode_dir, demo_name)
                else:
                    shutil.rmtree(recording_dir)
                    update_status(f"Demo {demo_num} discarded.")
                setattr(e, "demo_saved", bool(keep))
                raise
            except BaseException:
                self._preserve_partial(recording_dir, episode_dir, demo_name)
                raise

            if not controller_info["success"]:
                shutil.rmtree(recording_dir)
                update_status(f"Demo {demo_num} DISCARDED (B pressed). Resetting robot to re-record it...")
                self.env.reset()
                continue

            self._keep_demo(recording_dir, episode_dir, demo_name, demo_num, instruction, metadata, time.time() - start_time, timestamp, filename)
            update_status("Resetting robot...")
            self.env.reset()
            return

    def _record_trajectory(self, save_filepath: str, recording_folderpath: str, metadata: dict) -> dict:
        """Teleop until A or B is pressed, writing DROID's trajectory.h5 and SVO recordings.

        Mirrors droid.trajectory_utils.misc.collect_trajectory (with
        wait_for_controller=True), except the trajectory writer and camera
        recordings are always closed, so the files on disk are complete even if
        recording is interrupted or crashes.
        """
        env, controller = self.env, self.vr_controller
        controller.reset_state()
        env.camera_reader.set_trajectory_mode()

        traj_writer = TrajectoryWriter(save_filepath, metadata=metadata, save_images=False)
        env.camera_reader.start_recording(recording_folderpath)
        controller_info = {}
        try:
            while True:
                controller_info = controller.get_info()
                skip_action = not controller_info["movement_enabled"]
                control_timestamps = {"step_start": time_ms()}

                obs = env.get_observation()
                obs["controller_info"] = controller_info
                obs["timestamp"]["skip_action"] = skip_action

                control_timestamps["policy_start"] = time_ms()
                action, controller_action_info = controller.forward(obs, include_info=True)

                control_timestamps["sleep_start"] = time_ms()
                comp_time = time_ms() - control_timestamps["step_start"]
                sleep_left = (1 / env.control_hz) - (comp_time / 1000)
                if sleep_left > 0:
                    time.sleep(sleep_left)

                control_timestamps["control_start"] = time_ms()
                if skip_action:
                    action_info = env.create_action_dict(np.zeros_like(action))
                else:
                    action_info = env.step(action)
                action_info.update(controller_action_info)

                control_timestamps["step_end"] = time_ms()
                obs["timestamp"]["control"] = control_timestamps
                traj_writer.write_timestep({"observation": obs, "action": action_info})

                if controller_info["success"] or controller_info["failure"]:
                    return controller_info
        finally:
            with _defer_keyboard_interrupt():
                env.camera_reader.stop_recording()
                traj_writer.close(metadata=controller_info)

    def _keep_demo(self, recording_dir, episode_dir, demo_name, demo_num, instruction, metadata, duration, timestamp, filename) -> None:
        """Move a finished recording into demo_{n}/ (replacing any earlier take) and record it in the results file."""
        episode_id = uuid.uuid4().hex
        with open(os.path.join(recording_dir, f"metadata_{episode_id}.json"), "w") as f:
            json.dump({"uuid": episode_id, **metadata}, f, indent=2)

        demo_dir = os.path.join(episode_dir, demo_name)
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
        os.rename(recording_dir, demo_dir)

        timesteps = _trajectory_length(os.path.join(demo_dir, "trajectory.h5"))
        rollout_results = {
            "run_number": demo_num,
            "instruction": instruction,
            "duration": duration,
            "timesteps": timesteps,
            "data_files": [f"{demo_name}/{expected}" for expected in DEMO_EXPECTED_FILES],
            "timestamp": timestamp,
        }
        write_rollout_results(episode_dir, filename, demo_num, rollout_results)
        update_status(f"Demo {demo_num} SAVED ({timesteps} steps, {duration:.1f}s) -> {demo_dir}")

    def _preserve_partial(self, recording_dir, episode_dir, demo_name) -> None:
        """Keep an unfinished recording out of the dataset without deleting it."""
        if not os.path.exists(recording_dir):
            return
        base = os.path.join(episode_dir, f"{demo_name}_partial_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
        partial_dir, suffix = base, 1
        while os.path.exists(partial_dir):
            partial_dir, suffix = f"{base}_{suffix}", suffix + 1
        os.rename(recording_dir, partial_dir)
        trajectory = os.path.join(partial_dir, "trajectory.h5")
        if os.path.exists(trajectory):
            os.rename(trajectory, os.path.join(partial_dir, "trajectory_partial.h5"))
        update_status(f"Unfinished recording preserved (not counted as a demo) in {partial_dir}")

    def run_eval_loop(self, episode: EpisodePlanEntry, filename: str = DEMO_RESULTS_FILENAME, is_evaluation: bool = False) -> str:
        """Record every not-yet-saved demo for one episode, then an interactive
        retake loop until the user enters no demo numbers.
        """
        config_params = episode.episode_config
        episode_dir = episode.episode_dir

        demo_params = {
            "task_name": config_params["task_name"],
            "policy_name": TELEOP_POLICY_NAME,
            "policy_checkpoint": "",
            "instructions": config_params["instructions"],
            "num_rollouts": config_params["num_rollouts"],
            "max_step_score": config_params.get("max_step_score", 0),
            "max_recall_score": config_params.get("max_recall_score", 0),
            "max_timesteps": config_params.get("max_timesteps", 0),
            "folder_path": episode_dir,
            "expected_files": DEMO_EXPECTED_FILES,
        }
        write_eval_results(episode_dir, filename, demo_params)

        task_name = demo_params["task_name"]
        instructions = demo_params["instructions"]
        num_demos = demo_params["num_rollouts"]

        update_status("Starting demo collection for task: " + task_name)

        demo_batch = []
        for demo_num, status in enumerate(episode.rollout_statuses):
            if status == RolloutStatus.SCORE_DATA:
                print(f"[WARNING] Demo {demo_num} is missing data files -- re-recording it.")
            if status != RolloutStatus.VIDEO_DATA:
                demo_batch.append(demo_num)

        while True:
            remaining_batch = list(demo_batch)

            while remaining_batch:
                demo_num = remaining_batch[0]
                instruction = instructions[demo_num % len(instructions)]
                try:
                    self.record_demo(instruction, demo_num, num_demos, episode_dir, task_name, filename)
                    remaining_batch.pop(0)
                except KeyboardInterrupt as e:
                    if getattr(e, "demo_saved", False):
                        remaining_batch.pop(0)
                    choice = self._handle_episode_interrupt(is_evaluation)
                    if choice == "advance":
                        return "advance"
                    if choice == "quit":
                        raise
                    self.env.reset()

            update_status(f"All {num_demos} demos for {task_name} are saved in {episode_dir}")
            print("Enter demo numbers to re-record (whitespace-separated), or press Enter to finish:")
            retake_input = input("> ").strip()

            if not retake_input:
                return "done"

            try:
                retake_numbers = [int(token) for token in retake_input.split()]
            except ValueError:
                print(f"  Invalid input '{retake_input}' -- please enter whole numbers separated by whitespace.")
                continue

            out_of_range = [n for n in retake_numbers if n < 0 or n >= num_demos]
            if out_of_range:
                print(f"  Demo number(s) {out_of_range} are out of range [0, {num_demos}).")
                continue

            demo_batch = retake_numbers

    def run_evaluation_loop(self, plan: EvaluationPlan, filename: str = DEMO_RESULTS_FILENAME) -> None:
        super().run_evaluation_loop(plan, filename=filename)

    def run_test_loop(self) -> None:
        """Practice teleop without saving anything."""
        update_status("Practice mode: nothing is saved. Press A or B to end each run, Ctrl+C to stop.")

        try:
            while True:
                input("Press Enter to start a practice run...")
                collect_trajectory(self.env, controller=self.vr_controller, wait_for_controller=True, reset_robot=False)
                self.env.reset()
        except KeyboardInterrupt:
            pass



@contextlib.contextmanager
def _defer_keyboard_interrupt():
    """Delay Ctrl+C until the protected block finishes, so file cleanup can't be cut short."""
    interrupted = False

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    original_handler = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _trajectory_length(h5_path: str) -> int:
    """Number of timesteps in a DROID trajectory.h5 (every dataset has one row per step)."""
    lengths = []

    def visit(_name, obj):
        if isinstance(obj, h5py.Dataset) and not lengths:
            lengths.append(obj.shape[0])

    with h5py.File(h5_path, "r") as traj:
        traj.visititems(visit)
    return lengths[0] if lengths else 0
