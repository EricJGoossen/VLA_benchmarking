from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, ClassVar
from collections import deque
import functools
import numpy as np
import msgpack
import msgpack_numpy as mnp
import zmq

from third_party.openpi import image_tools
import json_numpy

json_numpy.patch()


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