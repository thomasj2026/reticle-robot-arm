from __future__ import annotations

from typing import Optional

import numpy as np
from robosuite.utils import transform_utils as T

from .robots import get_arm
from .tasks import make_raw_env, task_bddl, task_info, transfer_init_states

# Episode horizons used by OpenVLA / openpi for LIBERO evaluation.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
PANDA_FINGER_OPEN = 0.04
SETTLE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0])


def _flip(img: np.ndarray, mode: str) -> np.ndarray:
    # robosuite renders upside-down. "rot180" matches openpi / OpenVLA LIBERO data.
    if mode == "rot180":
        return np.ascontiguousarray(img[::-1, ::-1])
    if mode == "vertical":
        return np.ascontiguousarray(img[::-1])
    return img


class Libero6DofEnv:
    def __init__(
        self,
        suite: str = "libero_spatial",
        task_id: int = 0,
        robot: str = "UR5e",
        image_size: int = 256,
        image_flip: str = "rot180",
        max_steps: Optional[int] = None,
        settle_steps: int = 10,
        control_freq: int = 20,
        cameras: bool = True,
    ):
        get_arm(robot)  # registers Mounted<robot>/OnTheGround<robot>
        self.suite, self.task_id, self.robot = suite, task_id, robot
        self.info = task_info(suite, task_id)
        self.language = self.info["language"]
        self.image_flip = image_flip
        self.max_steps = max_steps or MAX_STEPS.get(suite, 400)
        self.settle_steps = settle_steps
        self.init_states = transfer_init_states(suite, task_id, robot)
        self.env = make_raw_env(
            task_bddl(suite, task_id), robot=robot, size=image_size, control_freq=control_freq,
            cameras=cameras,
        )
        self.cameras = cameras
        self._blank = np.zeros((image_size, image_size, 3), np.uint8)
        self._setup_gripper_map()
        self.t = 0
        self.episode_idx = None

    def _setup_gripper_map(self):
        self.env.reset()
        rb = self.env.env.robots[0]
        sim = self.env.sim
        jname = rb.gripper.joints[0]
        jid = sim.model.joint_name2id(jname)
        self._g_adr = sim.model.jnt_qposadr[jid]
        lo, hi = sim.model.jnt_range[jid]
        q_open = rb.gripper.init_qpos[0]
        self._g_open = q_open
        self._g_closed = hi if abs(hi - q_open) > abs(lo - q_open) else lo

    def gripper_open_fraction(self) -> float:
        q = self.env.sim.data.qpos[self._g_adr]
        frac = 1.0 - (q - self._g_open) / (self._g_closed - self._g_open)
        return float(np.clip(frac, 0.0, 1.0))


    def _obs(self, raw) -> dict:
        g = self.gripper_open_fraction()
        w = g * PANDA_FINGER_OPEN
        state = np.concatenate(
            [raw["robot0_eef_pos"], T.quat2axisangle(raw["robot0_eef_quat"]), [w, -w]]
        ).astype(np.float32)
        return {
            "image": _flip(raw["agentview_image"], self.image_flip) if self.cameras else self._blank,
            "wrist_image": _flip(raw["robot0_eye_in_hand_image"], self.image_flip) if self.cameras else self._blank,
            "state": state,
            "joint_pos": np.asarray(raw["robot0_joint_pos"], dtype=np.float32),
            "gripper_open": g,
            "prompt": self.language,
        }

    @property
    def num_init_states(self) -> int:
        return len(self.init_states)

    def reset(self, episode_idx: int = 0) -> dict:
        """Reset to LIBERO's fixed init state #episode_idx (object layout)."""
        self.env.reset()
        self.episode_idx = episode_idx % self.num_init_states
        raw = self.env.set_init_state(self.init_states[self.episode_idx])
        for _ in range(self.settle_steps):  # let objects settle, as in LIBERO eval
            raw, _, _, _ = self.env.step(SETTLE_ACTION)
        self.t = 0
        self._raw = raw
        return self._obs(raw)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(7), -1.0, 1.0)
        raw, reward, done, info = self.env.step(action)
        self.t += 1
        self._raw = raw
        success = bool(self.env.check_success())
        info = dict(info, success=success)
        done = success or self.t >= self.max_steps
        return self._obs(raw), float(success), done, info

    @property
    def raw_obs(self) -> dict:
        return self._raw

    @property
    def goal(self):
        """Parsed BDDL goal, e.g. [['on', 'akita_black_bowl_1', 'plate_1']]."""
        return self.env.env.parsed_problem["goal_state"]

    def object_pos(self, name: str) -> np.ndarray:
        e = self.env.env
        if name in e.object_states_dict:
            return np.array(e.object_states_dict[name].get_geom_state()["pos"])
        raise KeyError(name)

    def sim_state(self) -> np.ndarray:
        return self.env.sim.get_state().flatten()

    def render(self, size: Optional[int] = None) -> np.ndarray:
        return self._obs(self._raw)["image"]

    def close(self):
        self.env.close()
