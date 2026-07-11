from dataclasses import dataclass, field
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from .abstract_policy_client import AbstractPolicyClient


@dataclass
class OpenPiClient(AbstractPolicyClient, policy_name=["pi0", "pi05"]):
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
