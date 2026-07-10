import dataclasses

@dataclasses.dataclass
class Args:
    # Camera parameters
    scene_camera_id: str = "39668372"
    wrist_camera_id: str = "16744838"
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
 