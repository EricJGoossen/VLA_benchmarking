from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar
from collections import deque
import contextlib
import functools
import signal
import numpy as np
import msgpack
import msgpack_numpy as mnp
import requests
import zmq

from third_party.openpi import image_tools
from third_party.openpi import websocket_client_policy
import json_numpy

json_numpy.patch()


from eval_io import load_policy_config

@dataclass
class PolicyClient(ABC):
    """Base class for all policy clients."""

    action_space: str = ""
    gripper_space: str = ""
    policy_checkpoint: str = ""
    open_loop_horizon: int = 8
    host: str = "localhost"
    port: int = 8000
    name: str = ""

    # Class-level registry shared by all subclasses.
    _registry: ClassVar[dict[str, type["PolicyClient"]]] = {}

    def __init_subclass__(cls, *, policy_name: str | list[str] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if policy_name is not None:
            names = [policy_name] if isinstance(policy_name, str) else policy_name
            for name in names:
                PolicyClient._registry[name] = cls

    @classmethod
    def from_config(cls, config_path: str) -> "PolicyClient":
        """Instantiate the right client subclass from a loaded policy config dict."""
        config = load_policy_config(config_path)
        target_cls = cls._registry.get(config["policy_name"])
        if target_cls is None:
            raise ValueError(
                f"Unknown policy_name {config['policy_name']!r}. "
                f"Registered: {sorted(cls._registry)}"
            )

        kwargs = {k: v for k, v in config.items() if k != "policy_name"}

        valid_names = {f.name for f in fields(target_cls) if f.init}
        unknown = set(kwargs) - valid_names
        if unknown:
            raise ValueError(
                f"{target_cls.__name__} does not accept field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(valid_names)}"
            )

        return target_cls(**kwargs)

    def set_host(self, host: str) -> None:
        self.host = host

    def set_port(self, port: int) -> None:
        self.port = port

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def infer(self, observation: dict, instruction: str) -> np.ndarray:
        """Run inference and return a predicted action chunk (array of actions)."""
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @contextlib.contextmanager
    def prevent_keyboard_interrupt(self):
        """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
        interrupted = False
        original_handler = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if interrupted:
                raise KeyboardInterrupt


@dataclass
class OpenPiClient(PolicyClient, policy_name=["pi0", "pi05"]):
    """Client for pi0 / pi0.5 policies served via the openpi WebSocket server."""

    # Internal connection object — not a constructor arg, not shown in repr.
    _client: websocket_client_policy.WebsocketClientPolicy | None = field(
        default=None, init=False, repr=False
    )

    def connect(self):
        if self._client is None:
            self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def disconnect(self):
        self._client = None

    def infer(self, observation: dict, instruction: str) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("Client is not connected. Call connect() first.")

        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                observation["scene_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                observation["wrist_image"], 224, 224
            ),
            "observation/joint_position": observation["joint_position"],
            "observation/gripper_position": observation["gripper_position"],
            "prompt": instruction,
        }

        with self.prevent_keyboard_interrupt():
            try:
                actions = self._client.infer(request_data)["actions"]
                return np.clip(actions, -1, 1)
            except Exception:
                print("Disconnected — attempting to reconnect.")
                self.connect()
                actions = self._client.infer(request_data)["actions"]
                return np.clip(actions, -1, 1)


@dataclass
class MolmoActClient(PolicyClient, policy_name="molmoact"):
    """Client for MolmoAct policies served via an HTTP REST server."""

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def connect(self):
        try:
            response = requests.get(f"{self._base_url}/health", timeout=5)
            response.raise_for_status()
            print(f"MolmoAct server reachable at {self._base_url}")
        except requests.exceptions.ConnectionError:
            print(
                f"Warning: could not reach MolmoAct server at {self._base_url}. "
                "Continuing anyway — connection will be retried on first inference."
            )
        except requests.exceptions.HTTPError:
            pass  # Server is up but has no /health endpoint — that's fine.

    def disconnect(self):
        pass  # No persistent connection to close.

    def infer(self, observation: dict, instruction: str) -> np.ndarray:
        payload = json_numpy.dumps({
            "external_cam": observation["scene_image"],
            "wrist_cam": observation["wrist_image"],
            "instruction": instruction,
            "state": np.concatenate([observation["joint_position"], observation["gripper_position"]]),
        })

        with self.prevent_keyboard_interrupt():
            response = requests.post(
                f"{self._base_url}/act",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = json_numpy.loads(response.text)

        return result["actions"]


@dataclass
class _GrootConnection:
    """Internal connection state for GrootClient — not user-facing config."""

    context: zmq.Context = field(default_factory=zmq.Context)
    socket: "zmq.Socket | None" = None
    modality_keys: dict | None = None
    video_delta: list | None = None
    frame_buffer: deque | None = None


@dataclass
class GrootClient(PolicyClient, policy_name="gr00t"):
    """Client for GR00T policies served via a ZMQ server."""

    _EEF_ROTATION_CORRECT: ClassVar[np.ndarray] = np.array(
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
    )
    _IMAGE_RESOLUTION: ClassVar[tuple[int, int]] = (180, 320)  # (H, W)

    timeout_ms: int = 15000

    _conn: _GrootConnection = field(default_factory=_GrootConnection, init=False, repr=False)

    def connect(self):
        self._conn.socket = self._conn.context.socket(zmq.REQ)
        self._conn.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._conn.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._conn.socket.connect(f"tcp://{self.host}:{self.port}")

        self._send({"endpoint": "ping"})

        modality_config = self._send({"endpoint": "get_modality_config"})
        self._conn.modality_keys = {
            modality: modality_config[modality]["as_json"]["modality_keys"]
            for modality in ["video", "state", "action", "language"]
        }
        self._conn.video_delta = modality_config["video"]["as_json"]["delta_indices"]
        video_history_len = max(-min(self._conn.video_delta), 0) + 1 if self._conn.video_delta else 1
        self._conn.frame_buffer = deque(maxlen=video_history_len)

    def disconnect(self):
        if self._conn.socket is not None:
            self._conn.socket.close()
            self._conn.socket = None
        self._conn.modality_keys = None
        self._conn.video_delta = None
        self._conn.frame_buffer = None

    def infer(self, observation: dict, instruction: str) -> np.ndarray:
        if self._conn.socket is None or self._conn.modality_keys is None or self._conn.frame_buffer is None:
            raise RuntimeError("Client is not connected. Call connect() first.")

        H, W = self._IMAGE_RESOLUTION
        ext_image = image_tools.resize_with_pad(observation["scene_image"], H, W)
        wrist_image = image_tools.resize_with_pad(observation["wrist_image"], H, W)
        self._conn.frame_buffer.append({"ext": ext_image, "wrist": wrist_image})

        obs = self._format_observation(observation, instruction)

        with self.prevent_keyboard_interrupt():
            action_chunk, _ = self._send(
                {"endpoint": "get_action", "data": {"observation": obs, "options": None}}
            )

        return np.concatenate([
            action_chunk["joint_position"][0],
            action_chunk["gripper_position"][0],
        ], axis=-1)

    def _format_observation(self, observation: dict, instruction: str) -> dict:
        """Convert the standard observation dict into the nested format the GR00T server expects."""
        obs = {"video": {}, "state": {}, "language": {}}
        if self._conn.frame_buffer is None:
            raise RuntimeError("Frame buffer is not initialized.")
        if self._conn.modality_keys is None:
            raise RuntimeError("Modality keys are not initialized.")

        video_T = len(self._conn.video_delta) if self._conn.video_delta is not None else 1
        for key in self._conn.modality_keys["video"]:
            if video_T == 1:
                frame = self._conn.frame_buffer[-1]
                img = frame["wrist"] if "wrist" in key else frame["ext"]
                obs["video"][key] = img[None, None]
            else:
                hist = self._conn.frame_buffer[0]
                cur = self._conn.frame_buffer[-1]
                if hist is None or cur is None:
                    raise RuntimeError("Frame history contains None; cannot format video observation.")
                if "wrist" in key:
                    obs["video"][key] = np.stack([hist["wrist"], cur["wrist"]])[None]
                else:
                    obs["video"][key] = np.stack([hist["ext"], cur["ext"]])[None]

        state_source = {
            "eef_9d": self._compute_eef_9d(observation["cartesian_position"]),
            "gripper_position": observation["gripper_position"],
            "joint_position": observation["joint_position"],
        }
        for key in self._conn.modality_keys["state"]:
            val = state_source[key][None, None, ...].astype(np.float32)
            obs["state"][key] = val

        lang_key = self._conn.modality_keys["language"][0]
        obs["language"][lang_key] = [[instruction]]

        return obs

    @staticmethod
    def _compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
        """Convert XYZ + extrinsic XYZ Euler to XYZ + rot6d, corrected for OXE DROID convention."""
        from scipy.spatial.transform import Rotation
        c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
        rot_mat = Rotation.from_euler("XYZ", c[3:6]).as_matrix() @ GrootClient._EEF_ROTATION_CORRECT
        rot6d = rot_mat[:2, :].reshape(6)
        return np.concatenate([c[:3], rot6d]).astype(np.float32)

    def _send(self, request: dict):
        if self._conn.socket is None:
            raise RuntimeError("Client is not connected. Call connect() first.")
        self._conn.socket.send(GrootClient._MsgSerializer.to_bytes(request))
        response = GrootClient._MsgSerializer.from_bytes(self._conn.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    class _MsgSerializer:
        """msgpack + numpy serialization for ZMQ transport."""

        @staticmethod
        def to_bytes(data: Any) -> bytes:
            default = functools.partial(
                GrootClient._MsgSerializer._safe_encode,
                chain=GrootClient._MsgSerializer._encode_custom,
            )
            return msgpack.packb(data, default=default) or b""

        @staticmethod
        def from_bytes(data: bytes) -> Any:
            object_hook = functools.partial(
                GrootClient._MsgSerializer._safe_decode,
                chain=GrootClient._MsgSerializer._decode_custom,
            )
            return msgpack.unpackb(data, object_hook=object_hook, raw=False)

        @staticmethod
        def _safe_encode(obj, chain=None):
            if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
                raise TypeError(
                    f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                    "msgpack_numpy would invoke pickle."
                )
            return mnp.encode(obj, chain=chain)

        @staticmethod
        def _safe_decode(obj, chain=None):
            if isinstance(obj, dict):
                nd_val = obj.get(b"nd", obj.get("nd"))
                kind_val = obj.get(b"kind", obj.get("kind"))
                if nd_val and kind_val in (b"O", "O"):
                    raise ValueError("Refusing to decode object-dtype ndarray payload.")
            return mnp.decode(obj, chain=chain)

        @staticmethod
        def _encode_custom(obj):
            if is_dataclass(obj) and not isinstance(obj, type):
                return {"__dataclass__": type(obj).__name__, "fields": asdict(obj)}
            if isinstance(obj, Enum):
                return {"__enum__": type(obj).__name__, "value": obj.value}
            raise TypeError(f"Cannot encode object of type {type(obj)}")

        @staticmethod
        def _decode_custom(obj):
            if not isinstance(obj, dict):
                return obj
            if "__dataclass__" in obj or b"__dataclass__" in obj:
                key = next((k for k in ("fields", b"fields") if k in obj), None)
                if key is None:
                    raise ValueError("Malformed dataclass payload: 'fields' missing.")
                return obj[key]
            return obj