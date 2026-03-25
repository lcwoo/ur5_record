import datetime
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def _apply_camera_name_map(folder: Path, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Optionally rename camera observation keys using a user-editable map file.

    If `camera_name_map.json` exists in the recording folder, it should contain:
      {
        "cam0": "front_left",
        "cam1": "front_right",
        ...
      }

    Keys in obs like `cam0_rgb`, `cam0_depth` will be rewritten to
    `front_left_rgb`, `front_left_depth`.
    """
    map_path = folder / "camera_name_map.json"
    if not map_path.exists():
        return obs
    try:
        mapping = json.loads(map_path.read_text())
    except Exception:
        return obs
    if not isinstance(mapping, dict) or len(mapping) == 0:
        return obs

    out: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        new_k = k
        for src, dst in mapping.items():
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if k == f"{src}_rgb":
                new_k = f"{dst}_rgb"
                break
            if k == f"{src}_depth":
                new_k = f"{dst}_depth"
                break
        out[new_k] = v
    return out


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    # Work on a shallow copy so callers don't see in-place key changes.
    obs = dict(obs)

    # Optionally rename camera keys to user-defined names (hot-reload on every frame).
    obs = _apply_camera_name_map(folder, obs)

    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)
