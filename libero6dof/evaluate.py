from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .env import Libero6DofEnv
from .policies import RandomPolicy, RemotePolicy, ScriptedPickPlace, ZeroPolicy
from .tasks import get_suite


def run_episode(env, policy, episode_idx, replan_steps=5, video=None):
    obs = env.reset(episode_idx)
    policy.reset(env.language)
    frames = [np.concatenate([obs["image"], obs["wrist_image"]], 1)] if video else None
    queue, done, info = [], False, {"success": False}
    while not done:
        if not queue:
            a = np.asarray(policy.act(obs))
            chunk = a[None] if a.ndim == 1 else a
            queue = list(chunk[:replan_steps])
        obs, _, done, info = env.step(queue.pop(0))
        if frames is not None:
            frames.append(np.concatenate([obs["image"], obs["wrist_image"]], 1))
    if video:
        import imageio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=20)
    return bool(info["success"]), env.t


def make_policy(name, env, args):
    if name == "scripted":
        return ScriptedPickPlace(env)
    if name == "zero":
        return ZeroPolicy()
    if name == "random":
        return RandomPolicy()
    if name == "remote":
        return RemotePolicy(args.host, args.port, key_format=args.key_format, image_size=args.policy_image_size)
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all", help="'all' or comma list of task ids")
    ap.add_argument("--episodes", type=int, default=50, help="episodes per task (LIBERO has 50 fixed layouts)")
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--arm-config", default=None, help="YAML ArmSpec for a custom arm (overrides --robot)")
    ap.add_argument("--policy", default="scripted", choices=["scripted", "remote", "zero", "random"])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--key-format", default="openpi", choices=["openpi", "raw"])
    ap.add_argument("--policy-image-size", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--no-render", action="store_true", help="skip cameras (fast; only for privileged policies like scripted)")
    ap.add_argument("--videos-per-task", type=int, default=2)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args(argv)
    if a.arm_config:
        from .calibrate import load_spec
        from .robots import register_arm

        a.robot = register_arm(load_spec(a.arm_config)).name

    n_tasks = get_suite(a.suite).n_tasks
    task_ids = range(n_tasks) if a.tasks == "all" else [int(t) for t in a.tasks.split(",")]
    results, policy = {"suite": a.suite, "robot": a.robot, "policy": a.policy, "tasks": []}, None
    total_s = total_n = 0
    for tid in task_ids:
        env = Libero6DofEnv(
            a.suite, tid, robot=a.robot, image_size=a.image_size, max_steps=a.max_steps,
            cameras=not (a.no_render and not a.video_dir),
        )
        if policy is None or a.policy == "scripted":
            try:
                policy = make_policy(a.policy, env, a)
            except Exception as e:
                print(f"[task {tid}] cannot create policy: {e}")
                raise
        succ, t0 = [], time.time()
        for ep in range(a.episodes):
            video = None
            if a.video_dir and ep < a.videos_per_task:
                video = str(Path(a.video_dir) / f"{a.suite}_t{tid:02d}_ep{ep:02d}.mp4")
            try:
                ok, steps = run_episode(env, policy, ep, a.replan_steps, video)
            except ScriptedPickPlace.UnsupportedTask as e:
                print(f"[task {tid}] scripted expert does not support this goal ({e}); skipping")
                break
            succ.append(ok)
        env.close()
        if not succ:
            continue
        rate = float(np.mean(succ))
        total_s += sum(succ)
        total_n += len(succ)
        results["tasks"].append(
            {"task_id": tid, "language": env.language, "success_rate": rate, "episodes": len(succ)}
        )
        print(f"[task {tid:2d}] {rate*100:5.1f}%  ({sum(succ)}/{len(succ)})  {env.language}  [{time.time()-t0:.0f}s]")
    if hasattr(policy, "close"):
        policy.close()
    results["overall_success_rate"] = total_s / max(total_n, 1)
    print(f"\nOVERALL {a.suite} / {a.robot} / {a.policy}: {results['overall_success_rate']*100:.1f}% over {total_n} episodes")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
