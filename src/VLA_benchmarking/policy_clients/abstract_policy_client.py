from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar
import contextlib
import signal
import numpy as np

from ..eval_io import load_policy_config


@dataclass
class AbstractPolicyClient(ABC):
    """Base class for all policy clients."""

    action_space: str = ""
    gripper_space: str = ""
    policy_checkpoint: str = ""
    open_loop_horizon: int = 8
    host: str = "localhost"
    port: int = 8000
    policy_name: str = ""

    # Class-level registry shared by all subclasses.
    _registry: ClassVar[dict[str, type["AbstractPolicyClient"]]] = {}

    def __init_subclass__(cls, *, policy_name: str | list[str] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if policy_name is not None:
            names = [policy_name] if isinstance(policy_name, str) else policy_name
            for name in names:
                AbstractPolicyClient._registry[name] = cls

    @classmethod
    def from_config(cls, config_path: str) -> "AbstractPolicyClient":
        """Instantiate the right client subclass from a loaded policy config dict."""
        config = load_policy_config(config_path)
        target_cls = cls._registry.get(config["policy_name"])
        if target_cls is None:
            raise ValueError(
                f"Unknown policy_name {config['policy_name']!r}. "
                f"Registered: {sorted(cls._registry)}"
            )

        kwargs = {k: v for k, v in config.items() if k != "config_type"}

        if "default_remote_host" in kwargs:
            kwargs["host"] = kwargs.pop("default_remote_host")
        if "default_remote_port" in kwargs:
            kwargs["port"] = kwargs.pop("default_remote_port")

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
            