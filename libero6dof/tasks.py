from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
CACHE = Path(os.environ.get("LIBERO6DOF_CACHE", Path.home() / ".libero6dof"))
_ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")


def get_suite(suite: str):
    return benchmark.get_benchmark_dict()[suite]()


def task_bddl(suite: str, task_id: int) -> str:
    t = get_suite(suite).get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)


def task_info(suite: str, task_id: int) -> dict:
    t = get_suite(suite).get_task(task_id)
    return {"suite": suite, "task_id": task_id, "name": t.name, "language": t.language}


def list_tasks(suite: str) -> List[dict]:
    return [task_info(suite, i) for i in range(get_suite(suite).n_tasks)]


def make_raw_env(bddl: str, robot: str = "UR5e", cameras: bool = True, size: int = 256, **kw):
    """Plain LIBERO OffScreenRenderEnv with the chosen robot."""
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        robots=[robot],
        camera_heights=size,
        camera_widths=size,
        use_camera_obs=cameras,
        has_offscreen_renderer=cameras,
        **kw,
    )


def panda_init_states(suite: str, task_id: int) -> np.ndarray:
    # LIBERO's own loader calls torch.load(), which breaks on torch>=2.6
    # (weights_only default). These files ship with the LIBERO repo itself.
    import torch

    b = get_suite(suite)
    t = b.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file)
    return np.asarray(torch.load(path, weights_only=False))


def _scene_joints(sim):
    names = [sim.model.joint_id2name(j) for j in range(sim.model.njnt)]
    return [n for n in names if n and not n.startswith(_ROBOT_PREFIXES)]


def _joint_qpos_slice(sim, name):
    jid = sim.model.joint_name2id(name)
    adr = sim.model.jnt_qposadr[jid]
    width = {0: 7, 1: 4, 2: 1, 3: 1}[sim.model.jnt_type[jid]]  # free, ball, slide, hinge
    return slice(adr, adr + width)


def transfer_init_states(suite: str, task_id: int, robot: str, overwrite: bool = False) -> np.ndarray:
    """Convert LIBERO's Panda init states for one task into states for `robot`."""
    out = CACHE / "init_states" / robot / suite / f"task{task_id:02d}.npy"
    if out.exists() and not overwrite:
        return np.load(out)

    bddl = task_bddl(suite, task_id)
    src_states = panda_init_states(suite, task_id)
    src = make_raw_env(bddl, robot="Panda", cameras=False)
    dst = make_raw_env(bddl, robot=robot, cameras=False)
    src.reset()
    dst.reset()
    joints = _scene_joints(src.sim)
    missing = [j for j in joints if j not in _scene_joints(dst.sim)]
    if missing:
        raise RuntimeError(f"scene joints missing in {robot} env: {missing}")

    home = dst.sim.get_state()
    converted = []
    for s in src_states:
        src.set_init_state(s)
        dst.sim.set_state(home)
        for j in joints:
            dst.sim.data.qpos[_joint_qpos_slice(dst.sim, j)] = src.sim.data.qpos[_joint_qpos_slice(src.sim, j)]
        dst.sim.data.qvel[:] = 0
        dst.sim.forward()
        converted.append(dst.sim.get_state().flatten())
    src.close()
    dst.close()

    converted = np.stack(converted)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, converted)
    return converted
