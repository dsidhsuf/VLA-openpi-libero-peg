import json
from pathlib import Path
import numpy as np
from PIL import Image


def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z], dtype=np.float64) / s
    return (axis * angle).astype(np.float32)


def obs_to_state8(obs):
    eef_pos = np.asarray(obs['robot0_eef_pos'], dtype=np.float32)
    eef_quat = np.asarray(obs['robot0_eef_quat'], dtype=np.float32)
    eef_aa = quat_xyzw_to_axis_angle(eef_quat)
    gripper_qpos = np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32)
    return np.concatenate([eef_pos, eef_aa, gripper_qpos], axis=0).astype(np.float32)


def _resize_uint8_hwc(img: np.ndarray, image_size: int) -> np.ndarray:
    if img.shape[0] == image_size and img.shape[1] == image_size:
        return np.asarray(img, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
    pil = pil.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


class DemoRecorder:
    def __init__(self, save_dir, task_name, level, episode_id, task_text, image_size: int = 224, compressed: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.level = level
        self.episode_id = int(episode_id)
        self.task_text = task_text
        self.image_size = int(image_size)
        self.compressed = bool(compressed)
        self.agentview_images = []
        self.wrist_images = []
        self.states = []
        self.actions = []
        self.metrics = {}

    def record_step(self, obs, action):
        self.agentview_images.append(_resize_uint8_hwc(obs['agentview_image'], self.image_size))
        self.wrist_images.append(_resize_uint8_hwc(obs['robot0_eye_in_hand_image'], self.image_size))
        self.states.append(obs_to_state8(obs))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))

    def save(self, success, metrics=None):
        self.metrics = metrics or {}
        stem = f"{self.task_name}_{self.level}_ep{self.episode_id:04d}"
        npz_path = self.save_dir / f"{stem}.npz"
        json_path = self.save_dir / f"{stem}.json"

        arrays = dict(
            agentview_image=np.asarray(self.agentview_images, dtype=np.uint8),
            wrist_image=np.asarray(self.wrist_images, dtype=np.uint8),
            state8=np.asarray(self.states, dtype=np.float32),
            action=np.asarray(self.actions, dtype=np.float32),
            task=np.asarray([self.task_text] * len(self.actions)),
            success=np.asarray([bool(success)], dtype=np.bool_),
        )

        if self.compressed:
            np.savez_compressed(npz_path, **arrays)
        else:
            np.savez(npz_path, **arrays)

        meta = {
            'task_name': self.task_name,
            'level': self.level,
            'episode_id': self.episode_id,
            'task_text': self.task_text,
            'num_steps': len(self.actions),
            'success': bool(success),
            'metrics': self.metrics,
            'image_size': self.image_size,
            'compressed': self.compressed,
            'npz_path': str(npz_path),
        }
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print('[demo saved]', npz_path)
        print('[meta saved]', json_path)
