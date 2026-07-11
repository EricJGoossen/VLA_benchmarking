from dataclasses import dataclass
import numpy as np
import requests
import json_numpy
from .abstract_policy_client import AbstractPolicyClient


@dataclass
class MolmoActClient(AbstractPolicyClient, policy_name="molmoact"):
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
    