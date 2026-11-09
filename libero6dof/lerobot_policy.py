from __future__ import annotations

import os

import numpy as np


AGENTVIEW_KEY = "observation.images.image"
WRIST_KEY = "observation.images.image2"


def _rename_map(pipeline) -> dict:
    """The observation rename map baked into a saved preprocessor (e.g. image -> camera1)."""
    out = {}
    for step in getattr(pipeline, "steps", []):
        out.update(getattr(step, "rename_map", None) or {})
    return out


def resolve_cameras(policy_keys, rename_map):
    """Map each camera the policy expects to (key to feed, which sim camera).

    Fine-tuning SmolVLA keeps the base model's camera names (e.g. camera1/camera2)
    and stores a rename map in the preprocessor. We feed the dataset's names
    (image / image2) and let the preprocessor rename them. Without a rename map,
    cameras are matched by name, falling back to order (1st = agentview, 2nd = wrist).
    """
    inverse = {v: k for k, v in rename_map.items()}
    feed = {}
    for i, pk in enumerate(policy_keys):
        raw = inverse.get(pk, pk)
        if raw == AGENTVIEW_KEY:
            cam = "agentview"
        elif raw == WRIST_KEY:
            cam = "wrist"
        else:
            cam = "agentview" if i == 0 else "wrist"
        feed[raw] = (pk, cam)
    return feed


class LeRobotPolicy:
    def __init__(self, checkpoint: str, device: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = self.device
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=checkpoint, preprocessor_overrides={"device_processor": {"device": self.device}}
        )
        feats = cfg.input_features
        policy_keys = [k for k in feats if k.startswith("observation.images.")]
        self.feed = resolve_cameras(policy_keys, _rename_map(self.pre))
        self.image_hw = {raw: tuple(feats[pk].shape[-2:]) for raw, (pk, _) in self.feed.items()}
        desc = {raw: f"{cam} -> {pk} {self.image_hw[raw]}" for raw, (pk, cam) in self.feed.items()}
        print(f"loaded {cfg.type} from {checkpoint} on {self.device}; cameras {desc}")

    def _image(self, img: np.ndarray, key: str):
        import cv2

        h, w = self.image_hw[key]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    def act(self, obs: dict) -> np.ndarray:
        if "observation/image" in obs:  # openpi keys
            image, wrist = obs["observation/image"], obs["observation/wrist_image"]
            state, prompt = obs["observation/state"], obs["prompt"]
        else:  # libero6dof raw keys
            image, wrist, state, prompt = obs["image"], obs["wrist_image"], obs["state"], obs["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
        cams = {"agentview": image, "wrist": wrist}
        batch = {raw: self._image(np.asarray(cams[cam]), raw) for raw, (_, cam) in self.feed.items()}
        batch["observation.state"] = self.torch.as_tensor(np.asarray(state, np.float32)).unsqueeze(0)
        batch["task"] = [prompt]
        with self.torch.inference_mode():
            batch = self.pre(batch)
            self.policy.reset()  # the evaluator handles chunk execution; don't reuse queued actions
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.post(chunk)
        return np.asarray(chunk.squeeze(0).float().cpu().numpy(), dtype=np.float32)


def make_policy():
    ckpt = os.environ.get("LEROBOT_CHECKPOINT")
    if not ckpt:
        raise SystemExit("set LEROBOT_CHECKPOINT to a checkpoint folder or Hub repo id")
    return LeRobotPolicy(ckpt, os.environ.get("LEROBOT_DEVICE"))
