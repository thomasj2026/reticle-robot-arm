from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from .env import Libero6DofEnv
from .policies import ScriptedPickPlace
from .tasks import get_suite, task_bddl


def rollout(env, policy, ep_idx, noise=0.0, rng=None):
    obs = env.reset(ep_idx)
    policy.reset(env.language)
    buf = {k: [] for k in ("actions", "states", "rewards", "dones", "image", "wrist", "state", "joints")}
    done, info = False, {"success": False}
    while not done:
        a = np.asarray(policy.act(obs), dtype=np.float64)
        if noise and rng is not None:
            a[:6] = np.clip(a[:6] + rng.normal(0, noise, 6), -1, 1)
        buf["states"].append(env.sim_state())
        buf["image"].append(obs["image"])
        buf["wrist"].append(obs["wrist_image"])
        buf["state"].append(obs["state"])
        buf["joints"].append(obs["joint_pos"])
        buf["actions"].append(a)
        obs, r, done, info = env.step(a)
        buf["rewards"].append(r)
        buf["dones"].append(done)
    return bool(info["success"]), buf


def write_demo(grp, buf, ep_idx):
    n = len(buf["actions"])
    st = np.asarray(buf["state"])
    grp.attrs["num_samples"] = n
    grp.attrs["init_state_idx"] = ep_idx
    grp.create_dataset("actions", data=np.asarray(buf["actions"], np.float32))
    grp.create_dataset("states", data=np.asarray(buf["states"]))
    grp.create_dataset("rewards", data=np.asarray(buf["rewards"], np.float32))
    grp.create_dataset("dones", data=np.asarray(buf["dones"], np.uint8))
    o = grp.create_group("obs")
    o.create_dataset("agentview_rgb", data=np.asarray(buf["image"]), compression="gzip", compression_opts=4)
    o.create_dataset("eye_in_hand_rgb", data=np.asarray(buf["wrist"]), compression="gzip", compression_opts=4)
    o.create_dataset("ee_pos", data=st[:, :3])
    o.create_dataset("ee_ori", data=st[:, 3:6])
    o.create_dataset("ee_states", data=st[:, :6])
    o.create_dataset("gripper_states", data=st[:, 6:8])
    o.create_dataset("joint_states", data=np.asarray(buf["joints"], np.float32))


def collect_task(suite, tid, out_dir, robot, n_demos, max_attempts, noise, image_size, seed):
    env = Libero6DofEnv(suite, tid, robot=robot, image_size=image_size)
    policy = ScriptedPickPlace(env)
    rng = np.random.default_rng(seed + tid)
    path = Path(out_dir) / suite / f"task{tid:02d}_{env.info['name']}_demo.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = attempts = 0
    t0 = time.time()
    try:
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs.update(
                robot=robot, suite=suite, task_id=tid, language=env.language,
                bddl_file=task_bddl(suite, tid), image_flip=env.image_flip,
                problem_info=json.dumps({"language_instruction": env.language}),
                action_space="OSC_POSE delta [dx,dy,dz,droll,dpitch,dyaw,gripper(-1 open,+1 close)]",
            )
            while kept < n_demos and attempts < max_attempts:
                ep = attempts % env.num_init_states
                # first pass over the 50 layouts is noise-free, later passes add noise
                ep_noise = noise if attempts >= env.num_init_states else 0.0
                ok, buf = rollout(env, policy, ep, ep_noise, rng)
                attempts += 1
                if ok:
                    write_demo(data.create_group(f"demo_{kept}"), buf, ep)
                    kept += 1
            data.attrs["num_demos"] = kept
            data.attrs["attempts"] = attempts
    except ScriptedPickPlace.UnsupportedTask as e:
        path.unlink(missing_ok=True)
        print(f"[task {tid}] skipped: scripted expert can't do this goal ({e})")
        env.close()
        return None
    env.close()
    print(f"[task {tid:2d}] kept {kept}/{attempts}  {env.language}  -> {path}  [{time.time()-t0:.0f}s]")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--arm-config", default=None, help="YAML ArmSpec for a custom arm (overrides --robot)")
    ap.add_argument("--demos", type=int, default=50, help="successful demos to keep per task")
    ap.add_argument("--max-attempts", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.05, help="action noise std after the first 50 rollouts")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)
    if a.arm_config:
        from .calibrate import load_spec
        from .robots import register_arm

        a.robot = register_arm(load_spec(a.arm_config)).name
    n = get_suite(a.suite).n_tasks
    tids = range(n) if a.tasks == "all" else [int(t) for t in a.tasks.split(",")]
    for tid in tids:
        collect_task(a.suite, tid, a.out, a.robot, a.demos, a.max_attempts, a.noise, a.image_size, a.seed)


if __name__ == "__main__":
    main()
