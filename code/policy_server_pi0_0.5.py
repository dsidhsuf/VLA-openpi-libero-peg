import argparse
import base64
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

# ===== 兼容不同 lerobot 版本的 PI0 / PI05 导入 =====
_POLICY_CLASSES = []
_import_errors = []

for mod_path, cls_name in [
    ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    ("lerobot.policies.pi0", "PI0Policy"),
    ("lerobot.policies.pi0", "Pi0Policy"),
    ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    ("lerobot.policies.pi05", "PI05Policy"),
    ("lerobot.policies.pi05", "Pi05Policy"),
]:
    try:
        module = __import__(mod_path, fromlist=[cls_name])
        cls = getattr(module, cls_name)
        _POLICY_CLASSES.append((f"{mod_path}.{cls_name}", cls))
    except Exception as e:
        _import_errors.append(f"{mod_path}.{cls_name}: {repr(e)}")

if not _POLICY_CLASSES:
    raise ImportError(
        "Could not import any PI0/PI05 policy class. Tried:\n" + "\n".join(_import_errors)
    )

# 尝试拿到官方常量名；拿不到就用常见默认值
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

for mod_path in [
    "lerobot.policies.pi0.modeling_pi0",
    "lerobot.policies.pi05.modeling_pi05",
]:
    try:
        module = __import__(mod_path, fromlist=["OBS_LANGUAGE_TOKENS", "OBS_LANGUAGE_ATTENTION_MASK"])
        OBS_LANGUAGE_TOKENS = getattr(module, "OBS_LANGUAGE_TOKENS", OBS_LANGUAGE_TOKENS)
        OBS_LANGUAGE_ATTENTION_MASK = getattr(module, "OBS_LANGUAGE_ATTENTION_MASK", OBS_LANGUAGE_ATTENTION_MASK)
    except Exception:
        pass


SERVER_STATE = {
    "policy": None,
    "device": None,
    "policy_name": None,
    "input_keys": None,
    "state_dim": None,
    "tokenizer_source": None,
}


def decode_image_b64_to_numpy(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={img.shape}")
    return np.transpose(img, (2, 0, 1)).copy()


def get_input_feature_keys(policy):
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


def infer_state_dim_from_config(policy):
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

    return 8


def normalize_state_dim(state: np.ndarray, target_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == target_dim:
        return state
    if state.shape[0] > target_dim:
        return state[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[: state.shape[0]] = state
    return out


def find_text_tokenizer(policy):
    candidates = [
        ("policy.tokenizer", getattr(policy, "tokenizer", None)),
        ("policy.processor", getattr(policy, "processor", None)),
        ("policy.text_tokenizer", getattr(policy, "text_tokenizer", None)),
        ("policy.language_tokenizer", getattr(policy, "language_tokenizer", None)),
        ("policy.preprocessor", getattr(policy, "preprocessor", None)),
        ("policy.processor.tokenizer", getattr(getattr(policy, "processor", None), "tokenizer", None)),
        ("policy.preprocessor.tokenizer", getattr(getattr(policy, "preprocessor", None), "tokenizer", None)),
    ]

    for name, obj in candidates:
        if obj is not None:
            return name, obj
    return None, None


def tokenize_text(task_text: str, policy, device):
    name, tokenizer = find_text_tokenizer(policy)
    if tokenizer is None:
        raise RuntimeError(
            "No tokenizer/processor found on loaded policy. "
            "Need language tokenizer for PI0/PI05 inference."
        )

    SERVER_STATE["tokenizer_source"] = name

    attempts = [
        lambda: tokenizer([task_text], return_tensors="pt", padding=True, truncation=True),
        lambda: tokenizer(task_text, return_tensors="pt", padding=True, truncation=True),
        lambda: tokenizer(text=[task_text], return_tensors="pt", padding=True, truncation=True),
        lambda: tokenizer(text=task_text, return_tensors="pt", padding=True, truncation=True),
    ]

    last_err = None
    out = None
    for fn in attempts:
        try:
            out = fn()
            break
        except Exception as e:
            last_err = e

    if out is None:
        raise RuntimeError(f"Tokenizer call failed: {repr(last_err)}")

    # 常见 tokenizer 输出规范
    input_ids = None
    attn_mask = None

    if isinstance(out, dict):
        input_ids = out.get("input_ids", None) or out.get("tokens", None)
        attn_mask = out.get("attention_mask", None) or out.get("attention_masks", None)
    else:
        # 有些 processor 返回对象
        input_ids = getattr(out, "input_ids", None)
        attn_mask = getattr(out, "attention_mask", None)

    if input_ids is None:
        raise RuntimeError(
            f"Tokenizer output does not contain input_ids/tokens. type={type(out)} keys={list(out.keys()) if isinstance(out, dict) else 'N/A'}"
        )

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)
    elif not isinstance(attn_mask, torch.Tensor):
        attn_mask = torch.as_tensor(attn_mask)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attn_mask.ndim == 1:
        attn_mask = attn_mask.unsqueeze(0)

    return input_ids.to(device), attn_mask.to(device)


def build_batch_from_payload(payload, policy, device):
    input_keys = SERVER_STATE["input_keys"]
    state_dim = SERVER_STATE["state_dim"]

    raw_state = np.asarray(payload["observation.state"], dtype=np.float32)
    state = normalize_state_dim(raw_state, state_dim)

    image = decode_image_b64_to_numpy(payload["observation.images.image"])
    image2 = decode_image_b64_to_numpy(payload["observation.images.image2"])

    image_chw = hwc_to_chw(image)
    image2_chw = hwc_to_chw(image2)

    alias_pool = {
        "observation.state": state,
        "observation.images.image": image_chw,
        "observation.images.image2": image2_chw,
        "observation.images.base_0_rgb": image_chw,
        "observation.images.left_wrist_0_rgb": image2_chw,
        "observation.images.right_wrist_0_rgb": image2_chw,
    }

    batch = {}

    for k in input_keys:
        if k in alias_pool:
            arr = alias_pool[k]
            batch[k] = torch.from_numpy(np.asarray(arr)).to(device).unsqueeze(0)

    # 不管 input_keys 里显不显示，都补上语言 tokens
    task = payload.get("task", "")
    lang_tokens, lang_mask = tokenize_text(task, policy, device)
    batch[OBS_LANGUAGE_TOKENS] = lang_tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask

    # 同时保留原文本，便于个别版本内部还会读 task
    batch["task"] = [task]

    return batch


def load_policy(policy_path: str, device: torch.device):
    errors = []
    for name, cls in _POLICY_CLASSES:
        try:
            print(f"[server] trying loader: {name}")
            policy = cls.from_pretrained(policy_path)
            policy = policy.to(device)
            policy.eval()
            print(f"[server] loaded with: {name}")
            return policy, name
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    raise RuntimeError(
        "Failed to load policy from local path.\n"
        f"policy_path={policy_path}\n"
        "Tried loaders:\n" + "\n".join(errors)
    )


@torch.inference_mode()
def infer_action(payload):
    policy = SERVER_STATE["policy"]
    device = SERVER_STATE["device"]

    batch = build_batch_from_payload(payload, policy, device)

    if not hasattr(infer_action, "_printed"):
        print("[server] input_keys from checkpoint:", SERVER_STATE["input_keys"])
        print("[server] inferred state_dim:", SERVER_STATE["state_dim"])
        print("[server] tokenizer source:", SERVER_STATE["tokenizer_source"])
        print("[server] language keys:", OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        print("[server] batch keys sent to policy:", list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[server] {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        infer_action._printed = True

    action = policy.select_action(batch)

    if isinstance(action, torch.Tensor):
        action = action.squeeze(0).detach().cpu().numpy()
    elif isinstance(action, dict):
        if "action" in action:
            a = action["action"]
            if isinstance(a, torch.Tensor):
                action = a.squeeze(0).detach().cpu().numpy()
            else:
                action = np.asarray(a, dtype=np.float32)
        else:
            raise RuntimeError(f"Unknown dict output from select_action: {list(action.keys())}")
    else:
        action = np.asarray(action, dtype=np.float32)

    action = action.astype(np.float32).tolist()
    return action


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
                "policy_name": SERVER_STATE["policy_name"],
                "input_keys": SERVER_STATE["input_keys"],
                "state_dim": SERVER_STATE["state_dim"],
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
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[server] loading policy from local path: {args.policy_path}")
    print(f"[server] using device: {device}")

    policy, policy_name = load_policy(args.policy_path, device)

    input_keys = get_input_feature_keys(policy)
    state_dim = infer_state_dim_from_config(policy)

    SERVER_STATE["policy"] = policy
    SERVER_STATE["device"] = device
    SERVER_STATE["policy_name"] = policy_name
    SERVER_STATE["input_keys"] = input_keys
    SERVER_STATE["state_dim"] = state_dim

    print("[server] checkpoint input_keys:", input_keys)
    print("[server] checkpoint state_dim:", state_dim)

    httpd = HTTPServer((args.host, args.port), PolicyHandler)
    print(f"[server] listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
