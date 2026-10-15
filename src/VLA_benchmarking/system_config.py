import dataclasses

POLICY_CONFIG_KEYS = [
    "config_type",
    "policy_name",
    "policy_checkpoint",
    "open_loop_horizon",
    "default_remote_host",
    "default_remote_port",
    "action_space",
    "gripper_space",
]

EPISODE_CONFIG_KEYS = [
    "config_type",
    "task_name",
    "instructions",
    "max_timesteps",
    "num_rollouts",
    "max_step_score",
    "max_recall_score",
]

EVALUATION_CONFIG_KEYS = [
    "config_type",
    "evaluation_name",
    "episode_paths",
]

EVAL_RESULT_KEYS = [
    "task_name",
    "policy_name",
    "policy_checkpoint",
    "instructions",
    "num_rollouts",
    "max_step_score",
    "max_recall_score",
    "max_timesteps",
    "folder_path",
    "expected_files",
]

ROLLOUT_RESULT_KEYS = [
    "instruction",
    "duration",
    "timesteps",
    "run_number",
    "timestamp",
    "data_files",
]

SCORE_RESULT_KEYS = [
    "success",
    "step_score",
    "recall_score",
    "comments",
]

# Teleop demos are only written once the operator accepts them, so they carry no score fields
DEMO_SCORE_KEYS = []
DEMO_RESULTS_FILENAME = "demos.yaml"
DEMO_EXPECTED_FILES = ["trajectory.h5", "recordings"]

@dataclasses.dataclass
class Args:
    # Camera parameters
    scene_camera_id: str = "12345678" # UPDATE
    wrist_camera_id: str = "12345678" # UPDATE
    recording_fps: int = 10
    record_scene_camera: bool = True
    record_wrist_camera: bool = True

    # Configs
    policy_config: str = ""
    config_file: str = ""

    # Results
    results_dir: str = ""
    default_results_dir: str = "./results"

    # Server parameters
    server_host: str | None = None
    server_port: int | None = None

    # Teleop demo collection
    teleop: bool = False
    num_demos: int | None = None # Overrides num_rollouts from the task config when teleop is set
 