from __future__ import annotations

import argparse

import numpy as np
import yaml
from .robots import ArmSpec, get_arm, register_arm
from .tasks import make_raw_env, task_bddl


def _eef_pose(env):
    rb = env.env.robots[0]
    sid = rb.eef_site_id
    pos = np.array(env.sim.data.site_xpos[sid])
    mat = np.array(env.sim.data.site_xmat[sid]).reshape(3, 3)
    return pos, mat


def reference_pose(bddl: str):
    """End-effector pose of LIBERO's stock Panda at reset."""
    env = make_raw_env(bddl, robot="Panda", cameras=False)
    env.reset()
    pos, mat = _eef_pose(env)
    env.close()
    return pos, mat


def solve_ik(env, target_pos, target_mat, seed, iters=400, damping=1e-2):
    """Damped least-squares IK on the arm joints (gripper untouched)."""
    rb = env.env.robots[0]
    sim = env.sim
    qidx = rb._ref_joint_pos_indexes
    vidx = rb._ref_joint_vel_indexes
    site = rb.gripper.important_sites["grip_site"]
    lo, hi = sim.model.jnt_range[rb._ref_joint_indexes].T
    q = np.array(seed, dtype=float)
    for _ in range(iters):
        sim.data.qpos[qidx] = q
        sim.forward()
        pos, mat = _eef_pose(env)
        e_pos = target_pos - pos
        e_rot = 0.5 * sum(np.cross(mat[:, i], target_mat[:, i]) for i in range(3))
        err = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
            break
        Jp = sim.data.get_site_jacp(site).reshape(3, -1)[:, vidx]
        Jr = sim.data.get_site_jacr(site).reshape(3, -1)[:, vidx]
        J = np.vstack([Jp, Jr])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)
    return q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))


def calibrate(spec: ArmSpec, bddl: str | None = None, verbose=True):
    bddl = bddl or task_bddl("libero_spatial", 0)
    target_pos, target_mat = reference_pose(bddl)
    register_arm(spec)
    env = make_raw_env(bddl, robot=spec.name, cameras=False)
    env.reset()
    seeds = [np.array(spec.home_qpos)]
    rng = np.random.default_rng(0)
    seeds += [np.array(spec.home_qpos) + rng.normal(0, 0.6, spec.dof) for _ in range(15)]
    best = None
    for s in seeds:
        q, ep, er = solve_ik(env, target_pos, target_mat, s)
        # prefer accurate solutions, then ones close to the nominal posture
        score = ep * 100 + er * 10 + 0.01 * np.linalg.norm(q - spec.home_qpos)
        if best is None or score < best[0]:
            best = (score, q, ep, er)
    env.close()
    _, q, ep, er = best
    if verbose:
        print(f"target eef pos {target_pos.round(4)}")
        print(f"home_qpos: [{', '.join(f'{v:.5f}' for v in q)}]")
        print(f"residual: pos {ep*1000:.2f} mm, rot {np.degrees(er):.3f} deg")
    return q, ep, er


def load_spec(path: str) -> ArmSpec:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ArmSpec(**cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="UR5e", help="Built-in arm name")
    ap.add_argument("--arm-config", help="YAML ArmSpec for a custom arm")
    ap.add_argument("--write", action="store_true", help="Write home_qpos back to --arm-config")
    a = ap.parse_args()
    spec = load_spec(a.arm_config) if a.arm_config else get_arm(a.arm)
    q, ep, er = calibrate(spec)
    if ep > 0.005 or er > np.radians(1):
        print("WARNING: IK did not converge well; the home pose may be outside the arm's reach.")
    if a.write and a.arm_config:
        with open(a.arm_config) as f:
            cfg = yaml.safe_load(f)
        cfg["home_qpos"] = [round(float(v), 5) for v in q]
        with open(a.arm_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote home_qpos to {a.arm_config}")


if __name__ == "__main__":
    main()
