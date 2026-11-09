from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

FPS = 20  # libero6dof / robosuite control frequency

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "finger_l", "finger_r"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def features(h: int, w: int, use_videos: bool) -> dict:
    img = {
        "dtype": "video" if use_videos else "image",
        "shape": (h, w, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.image": dict(img),
        "observation.images.image2": dict(img),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def demo_files(src: Path):
    files = sorted(p for p in src.rglob("*_demo.hdf5") if p.stat().st_size > 1024)
    if not files:
        raise SystemExit(f"no *_demo.hdf5 files with demos under {src}")
    return files


def convert(src, repo_id, root=None, use_videos=True, push=False, private=False, overwrite=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = demo_files(Path(src))
    with h5py.File(files[0]) as f:
        first = f["data"][sorted(f["data"].keys())[0]]
        h, w = first["obs/agentview_rgb"].shape[1:3]
        robot = f["data"].attrs.get("robot", "UR5e")

    if root and Path(root).exists():
        if not overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features(h, w, use_videos),
        root=root,
        robot_type=f"libero6dof_{robot}",
        use_videos=use_videos,
        image_writer_threads=4,
    )

    n_eps = 0
    for path in files:
        with h5py.File(path) as f:
            data = f["data"]
            task = str(data.attrs["language"])
            for name in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                g = data[name]
                imgs = g["obs/agentview_rgb"][()]
                wrist = g["obs/eye_in_hand_rgb"][()]
                state = np.concatenate([g["obs/ee_states"][()], g["obs/gripper_states"][()]], axis=1)
                actions = g["actions"][()]
                if imgs.shape[1:3] != (h, w):
                    raise SystemExit(f"{path}: image size {imgs.shape[1:3]} != {(h, w)}; collect all demos at one size")
                for t in range(len(actions)):
                    ds.add_frame(
                        {
                            "observation.images.image": imgs[t],
                            "observation.images.image2": wrist[t],
                            "observation.state": state[t].astype(np.float32),
                            "action": actions[t].astype(np.float32),
                            "task": task,
                        }
                    )
                ds.save_episode()
                n_eps += 1
        print(f"{path.name}: done ({n_eps} episodes so far)")

    ds.finalize()
    print(f"dataset {repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames at {ds.root}")
    if push:
        ds.push_to_hub(private=private)
        print(f"pushed to https://huggingface.co/datasets/{repo_id}")
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder containing *_demo.hdf5 (searched recursively)")
    ap.add_argument("--repo-id", required=True, help="<hf_user>/<dataset_name>")
    ap.add_argument("--root", default=None, help="local output folder (default: LeRobot cache)")
    ap.add_argument("--images", action="store_true", help="store PNG frames instead of MP4 video")
    ap.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    convert(a.src, a.repo_id, a.root, not a.images, a.push, a.private, a.overwrite)


if __name__ == "__main__":
    main()
