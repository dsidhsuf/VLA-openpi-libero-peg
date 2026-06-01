from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class CustomTaskSpec:
    """A lightweight task spec compatible with LIBERO-style evaluation loops."""

    name: str
    language: str
    bddl_file: str
    init_states_file: str
    max_steps: int = 300
    num_wait_steps: int = 10
    success: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"type": "env_check_success"}
    )


class CustomLiberoSuite:
    """
    Runtime task suite defined by JSON, with explicit fixed init states.

    JSON schema:
    {
      "suite_name": "my_suite",
      "tasks": [
        {
          "name": "...",
          "language": "...",
          "bddl_file": "/abs/path/to/task.bddl",
          "init_states_file": "/abs/path/to/init_states.pt",
          "max_steps": 360,
          "num_wait_steps": 10,
          "success": { "type": "pose_threshold", ... }
        }
      ]
    }
    """

    def __init__(self, suite_name: str, tasks: list[CustomTaskSpec]):
        if not tasks:
            raise ValueError("CustomLiberoSuite requires at least one task.")
        self.suite_name = suite_name
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self._init_states_cache: dict[int, np.ndarray] = {}

    def get_task(self, i: int) -> CustomTaskSpec:
        self._validate_task_index(i)
        return self.tasks[i]

    def get_task_init_states(self, i: int) -> np.ndarray:
        self._validate_task_index(i)
        if i not in self._init_states_cache:
            path = pathlib.Path(self.tasks[i].init_states_file).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Init states file does not exist: {path}")
            self._init_states_cache[i] = _load_init_states(path)
        return self._init_states_cache[i]

    def _validate_task_index(self, i: int) -> None:
        if i < 0 or i >= self.n_tasks:
            raise IndexError(f"Task index out of range: {i} (n_tasks={self.n_tasks})")

    @classmethod
    def from_json(cls, json_path: str | pathlib.Path) -> "CustomLiberoSuite":
        path = pathlib.Path(json_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Suite json does not exist: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        suite_name = str(raw.get("suite_name", path.stem))
        raw_tasks = raw.get("tasks", [])
        if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
            raise ValueError("Suite json must contain a non-empty 'tasks' list.")

        tasks: list[CustomTaskSpec] = []
        for idx, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                raise TypeError(f"tasks[{idx}] must be a JSON object.")

            for key in ("name", "language", "bddl_file", "init_states_file"):
                if key not in item:
                    raise KeyError(f"tasks[{idx}] missing required key: '{key}'")

            bddl_file = _resolve_path(path.parent, str(item["bddl_file"]))
            init_states_file = _resolve_path(path.parent, str(item["init_states_file"]))

            tasks.append(
                CustomTaskSpec(
                    name=str(item["name"]),
                    language=str(item["language"]),
                    bddl_file=bddl_file,
                    init_states_file=init_states_file,
                    max_steps=int(item.get("max_steps", 300)),
                    num_wait_steps=int(item.get("num_wait_steps", 10)),
                    success=dict(item.get("success", {"type": "env_check_success"})),
                )
            )
        return cls(suite_name=suite_name, tasks=tasks)


def _resolve_path(base_dir: pathlib.Path, path_str: str) -> str:
    candidate = pathlib.Path(path_str).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base_dir / candidate).resolve())


def _load_init_states(path: pathlib.Path) -> np.ndarray:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        if "init_states" not in loaded:
            raise KeyError(
                f"Init states dict at {path} must contain key 'init_states'. "
                f"Found keys={list(loaded.keys())}"
            )
        loaded = loaded["init_states"]

    if torch.is_tensor(loaded):
        arr = loaded.detach().cpu().numpy()
    elif isinstance(loaded, np.ndarray):
        arr = loaded
    elif isinstance(loaded, list):
        states = [np.asarray(x, dtype=np.float64).reshape(-1) for x in loaded]
        if not states:
            raise ValueError(f"Empty init states list in {path}")
        arr = np.stack(states, axis=0)
    else:
        raise TypeError(
            f"Unsupported init states format in {path}: {type(loaded).__name__}"
        )

    if arr.ndim != 2:
        raise ValueError(
            f"Init states should have shape [N, D], got shape {arr.shape} from {path}"
        )
    return np.asarray(arr, dtype=np.float64)
