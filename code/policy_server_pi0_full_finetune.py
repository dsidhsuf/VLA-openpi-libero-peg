import argparse
import base64
import io
import json
import os
import shutil
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

from lerobot.policies.factory import make_pre_post_processors

try:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
except Exception:
    try:
        from lerobot.policies.pi0 import PI0Policy
    except Exception:
        from lerobot.policies.pi0 import Pi0Policy as PI0Policy


SERVER_STATE = {
    "policy": None,
    "device": None,
    "preprocess": None,
    "postprocess": None,
    "policy_path": None,
    "tokenizer_path": None,
    "runtime_model_dir": None,
    "input_keys": None,
    "state_dim": None,
    "return_chunk": True,
    "chunk_len": None,
    "rotate_images_180": False,
}


def decode_image_b64_to_numpy(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def rotate_180_hwc(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, 2, axes=(0, 1)).copy()


def normalize_state_dim(state: np.ndarray, target_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == target_dim:
        return state
    if state.shape[0] > target_dim:
        return state[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[: state.shape[0]] = state
    return out


def infer_input_keys(policy):
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return []

    for attr in ["input_features", "observation_features"]:
        feats = getattr(cfg, attr, None)
        if feats is not None:
            try:
                return list(feats.keys())
            except Exception:
                pass
    return []


def infer_state_dim(policy):
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return 8

    for attr in ["input_features", "observation_features"]:
        feats = getattr(cfg, attr, None)
        if feats is None:
            continue
        if "observation.state" in feats:
            feat = feats["observation.state"]
            shape = getattr(feat, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])

    max_state_dim = getattr(cfg, "max_state_dim", None)
    if max_state_dim is not None:
        return int(max_state_dim)

    return 8


def link_all(src_dir: str, dst_dir: str):
    if src_dir is None:
        return
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def patch_processor_jsons(runtime_dir: str, tokenizer_path: str):
    for fname in ["policy_preprocessor.json", "policy_postprocessor.json", "preprocessor_config.json"]:
        fpath = os.path.join(runtime_dir, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        changed = False

        def _walk(x):
            nonlocal changed
            if isinstance(x, dict):
                for k, v in list(x.items()):
                    if k in ["tokenizer_name", "processor_name", "pretrained_model_name_or_path"]:
                        if isinstance(v, str) and v == "google/paligemma-3b-pt-224":
                            x[k] = tokenizer_path
                            changed = True
                    else:
                        _walk(v)
            elif isinstance(x, list):
                for item in x:
                    _walk(item)

        _walk(obj)

        if changed:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            print(f"[server] patched processor config: {fpath}")


def build_runtime_model_dir(policy_path: str, tokenizer_path: str) -> str:
    runtime_dir = os.path.join(tempfile.gettempdir(), "pi0_libero_runtime_model")

    if os.path.exists(runtime_dir):
        shutil.rmtree(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)

    link_all(policy_path, runtime_dir)
    if tokenizer_path and os.path.abspath(tokenizer_path) != os.path.abspath(policy_path):
        link_all(tokenizer_path, runtime_dir)

    patch_processor_jsons(runtime_dir, tokenizer_path)
    return runtime_dir


def build_raw_frame_from_payload(payload, state_dim: int):
    raw_state = np.asarray(payload["observation.state"], dtype=np.float32)
    state = normalize_state_dim(raw_state, state_dim)

    image = decode_image_b64_to_numpy(payload["observation.images.image"])
    image2 = decode_image_b64_to_numpy(payload["observation.images.image2"])

    # The eval client already sends images in the same orientation as the
    # recorded LeRobot videos. Keep this off unless you intentionally use an
    # older client that does not flip images before sending them.
    if SERVER_STATE["rotate_images_180"]:
        image = rotate_180_hwc(image)
        image2 = rotate_180_hwc(image2)

    empty_camera = np.zeros_like(image, dtype=np.uint8)

    task = payload.get("task", "")

    return {
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.images.empty_camera_0": empty_camera,
        "observation.state": state,
        "task": task,
        "task_description": task,
        "language_instruction": task,
    }


def prepare_batch_for_policy(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, str):
            out[k] = v
            continue
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
            out[k] = list(v)
            continue

        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            t = v
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype.kind in ["U", "S", "O"]:
                    out[k] = v
                    continue
                t = torch.from_numpy(arr)
            except Exception:
                out[k] = v
                continue
        else:
            out[k] = v
            continue

        # Normalize image tensors to BCHW.
        if k.startswith("observation.images."):
            if t.ndim == 3:
                # HWC -> CHW if needed, then add batch dimension.
                if t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1).contiguous()
                t = t.unsqueeze(0)
            elif t.ndim == 4 and t.shape[-1] in (1, 3):
                # BHWC -> BCHW
                t = t.permute(0, 3, 1, 2).contiguous()

        elif k == "observation.state" and t.ndim == 1:
            t = t.unsqueeze(0)
        elif "language" in k and t.ndim == 1:
            t = t.unsqueeze(0)
        elif "attention_mask" in k and t.ndim == 1:
            t = t.unsqueeze(0)

        out[k] = t.to(device)

    return out


def tensor_or_action_dict_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict) and "action" in value:
        action = value["action"]
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy()
        return np.asarray(action)
    return np.asarray(value)


def postprocess_action_array(action_tensor: torch.Tensor, postprocess) -> np.ndarray:
    """Post-process normalized actions while preserving a possible time axis."""
    original_shape = tuple(action_tensor.shape)
    if action_tensor.ndim == 3:
        batch, horizon, action_dim = action_tensor.shape
        flat = action_tensor.reshape(batch * horizon, action_dim)
        processed = postprocess(flat)
        arr = tensor_or_action_dict_to_numpy(processed)
        arr = np.asarray(arr, dtype=np.float32).reshape(batch, horizon, -1)
        return arr

    processed = postprocess(action_tensor)
    arr = tensor_or_action_dict_to_numpy(processed)
    arr = np.asarray(arr, dtype=np.float32)
    if len(original_shape) == 2 and arr.ndim == 1:
        arr = arr.reshape(original_shape[0], -1)
    return arr


@torch.inference_mode()
def infer_action(payload):
    policy = SERVER_STATE["policy"]
    preprocess = SERVER_STATE["preprocess"]
    postprocess = SERVER_STATE["postprocess"]
    state_dim = SERVER_STATE["state_dim"]

    raw_frame = build_raw_frame_from_payload(payload, state_dim)
    batch = preprocess(raw_frame)
    batch = prepare_batch_for_policy(batch, SERVER_STATE["device"])

    if not hasattr(infer_action, "_printed"):
        print("[server] input_keys from checkpoint:", SERVER_STATE["input_keys"])
        print("[server] inferred state_dim:", SERVER_STATE["state_dim"])
        print("[server] runtime_model_dir:", SERVER_STATE["runtime_model_dir"])
        print("[server] preprocessed batch keys:", list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[server] {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        infer_action._printed = True

    if SERVER_STATE["return_chunk"] and hasattr(policy, "predict_action_chunk"):
        pred_action = policy.predict_action_chunk(batch)
        action = postprocess_action_array(pred_action, postprocess)

        if action.ndim != 3 or action.shape[0] != 1:
            raise RuntimeError(
                f"Expected action chunk shape (1, T, 7) after postprocess, got shape={tuple(action.shape)}"
            )

        action = action[0]
        target_chunk_len = SERVER_STATE["chunk_len"]
        if target_chunk_len is not None and int(target_chunk_len) > 0:
            action = action[: int(target_chunk_len)]

        if action.ndim != 2 or action.shape[1] < 7:
            raise RuntimeError(
                f"Expected action chunk shape (T, >=7) after postprocess, got shape={tuple(action.shape)}"
            )

        action = action[:, :7].astype(np.float32)
        return action.tolist()

    pred_action = policy.select_action(batch)
    action = postprocess_action_array(pred_action, postprocess)

    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    elif action.ndim != 1:
        action = action.reshape(-1)

    if action.shape[0] < 7:
        raise RuntimeError(
            f"Expected at least 7-dim action from pi0_libero_base after postprocess, got shape={tuple(action.shape)}"
        )

    action = action[:7]
    return action.astype(np.float32).tolist()


class PolicyHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "policy_path": SERVER_STATE["policy_path"],
                "tokenizer_path": SERVER_STATE["tokenizer_path"],
                "runtime_model_dir": SERVER_STATE["runtime_model_dir"],
                "input_keys": SERVER_STATE["input_keys"],
                "state_dim": SERVER_STATE["state_dim"],
                "return_chunk": SERVER_STATE["return_chunk"],
                "chunk_len": SERVER_STATE["chunk_len"],
                "rotate_images_180": SERVER_STATE["rotate_images_180"],
            })
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}

            if self.path == "/reset":
                policy = SERVER_STATE["policy"]
                if hasattr(policy, "reset"):
                    try:
                        policy.reset()
                    except Exception:
                        pass
                self._send_json({"status": "reset"})
                return

            if self.path == "/infer":
                action = infer_action(payload)
                self._send_json({"action": action})
                return

            self._send_json({"error": "not found"}, code=404)

        except Exception as e:
            tb = traceback.format_exc()
            print("[server] infer exception:")
            print(tb)
            self._send_json({
                "error": repr(e),
                "traceback": tb,
            }, code=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--single_action",
        action="store_true",
        help="Return only one action via policy.select_action() instead of a full action chunk.",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=None,
        help="Max number of actions to return from predict_action_chunk(). Default uses checkpoint chunk_size.",
    )
    parser.add_argument(
        "--rotate_images_180",
        action="store_true",
        help="Legacy compatibility: rotate decoded images by 180 degrees inside the server.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy_path = args.policy_path
    tokenizer_path = args.tokenizer_path or args.policy_path

    runtime_model_dir = build_runtime_model_dir(policy_path, tokenizer_path)

    print(f"[server] loading pi0_libero_base from: {policy_path}")
    print(f"[server] tokenizer path: {tokenizer_path}")
    print(f"[server] runtime model dir: {runtime_model_dir}")
    print(f"[server] using device: {device}")

    policy = PI0Policy.from_pretrained(runtime_model_dir).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        runtime_model_dir,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    input_keys = infer_input_keys(policy)
    state_dim = infer_state_dim(policy)

    SERVER_STATE["policy"] = policy
    SERVER_STATE["device"] = device
    SERVER_STATE["preprocess"] = preprocess
    SERVER_STATE["postprocess"] = postprocess
    SERVER_STATE["policy_path"] = policy_path
    SERVER_STATE["tokenizer_path"] = tokenizer_path
    SERVER_STATE["runtime_model_dir"] = runtime_model_dir
    SERVER_STATE["input_keys"] = input_keys
    SERVER_STATE["state_dim"] = state_dim
    SERVER_STATE["return_chunk"] = not args.single_action
    SERVER_STATE["chunk_len"] = args.chunk_len
    SERVER_STATE["rotate_images_180"] = bool(args.rotate_images_180)

    print("[server] checkpoint input_keys:", input_keys)
    print("[server] checkpoint state_dim:", state_dim)
    print("[server] return_chunk:", SERVER_STATE["return_chunk"])
    print("[server] chunk_len:", SERVER_STATE["chunk_len"])
    print("[server] rotate_images_180:", SERVER_STATE["rotate_images_180"])

    httpd = HTTPServer((args.host, args.port), PolicyHandler)
    print(f"[server] listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
