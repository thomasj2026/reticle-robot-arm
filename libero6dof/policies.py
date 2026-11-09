from __future__ import annotations

import numpy as np
from robosuite.utils import transform_utils as T


class Policy:
    def reset(self, prompt: str) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class ZeroPolicy(Policy):
    """Holds still with the gripper open. Useful as a smoke test."""

    def act(self, obs):
        return np.array([0, 0, 0, 0, 0, 0, -1.0])


class RandomPolicy(Policy):
    def __init__(self, scale=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale

    def act(self, obs):
        a = self.rng.uniform(-self.scale, self.scale, 7)
        a[6] = np.sign(a[6]) or -1.0
        return a


# --------------------------------------------------------------------------- #
# Scripted expert (uses privileged simulator state)                            #
# --------------------------------------------------------------------------- #
class ScriptedPickPlace(Policy):
    """Waypoint expert for single "pick X, place on/in Y" LIBERO goals.

    Covers libero_spatial, libero_object and the pick-place subset of
    libero_goal / libero_90. It reads object poses from the simulator, so it is a
    data generator and a reachability check, not a learned policy.
    Raises ``UnsupportedTask`` for goals it can't handle (open drawer, turn on
    stove, multi-step goals).
    """

    class UnsupportedTask(Exception):
        pass

    POS_GAIN = 12.0
    ROT_GAIN = 3.0
    SMALL_OBJECT = 0.078  # m; wider than this along the closing axis -> rim grasp (2F-85 opens 85 mm)
    RIM_INSET = 0.006  # m from the outer edge to the grasp centre on a rim
    RIM_DEPTH = 0.046  # m below the top of the object for rim grasps (clamped to 85% of height)
    CENTER_DEPTH = 0.035  # m below the top for centre grasps
    CLOSE_STEPS = 25

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose

    # --- scene geometry helpers -------------------------------------------
    def _sim(self):
        return self.env.env.sim

    def _subtree_geoms(self, body_id):
        m = self._sim().model._model
        bodies = {body_id}
        for b in range(m.nbody):
            p = b
            while p != 0:
                if p == body_id:
                    bodies.add(b)
                    break
                p = m.body_parentid[p]
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]

    def _object_aabb(self, name):
        """World-frame axis-aligned bounds of an object's geoms."""
        e = self.env.env.env
        sim = self._sim()
        m, d = sim.model._model, sim.data._data
        # Use visual + collision geoms: LIBERO's convex decompositions can be
        # noticeably smaller than the visible object (e.g. the black bowl).
        geoms = self._subtree_geoms(e.obj_body_id[name]) if name in e.obj_body_id else []
        if not geoms:  # a region site: treat as a thin pad at the site
            p = self.env.object_pos(name)
            return p - [0.03, 0.03, 0.0], p + [0.03, 0.03, 0.0]
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            if m.geom_type[g] == 7:  # mesh: use its vertices (a rotated aabb inflates bounds)
                mid = m.geom_dataid[g]
                pts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
            w = pts @ R.T + d.geom_xpos[g]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        return lo, hi

    def _closing_axis(self):
        """Unit vector (world xy) along which the fingers close."""
        rb = self.env.env.env.robots[0]
        sim = self._sim()
        geoms = rb.gripper.important_geoms
        left = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(g)] for g in geoms["left_fingerpad"]], 0)
        right = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(g)] for g in geoms["right_fingerpad"]], 0)
        v = (left - right)[:2]
        return np.append(v / (np.linalg.norm(v) + 1e-9), 0.0)

    def _pad_offset(self):
        """Height of the finger-pad centres above the TCP site (gripper specific)."""
        rb = self.env.env.env.robots[0]
        sim = self._sim()
        geoms = rb.gripper.important_geoms
        pads = [sim.data.geom_xpos[sim.model.geom_name2id(g)] for g in geoms["left_fingerpad"] + geoms["right_fingerpad"]]
        site = sim.data.site_xpos[rb.eef_site_id]
        return float(np.mean(pads, 0)[2] - site[2])

    # --- planning ----------------------------------------------------------
    def _plan(self):
        goal = self.env.goal
        if len(goal) != 1 or len(goal[0]) != 3 or goal[0][0].lower() not in ("on", "in"):
            raise self.UnsupportedTask(f"goal {goal}")
        _, obj, target = goal[0]
        olo, ohi = self._object_aabb(obj)
        tlo, thi = self._object_aabb(target)
        ocen = (olo + ohi) / 2
        size = ohi - olo
        axis = self._closing_axis()
        # For compact objects, turn the wrist 90 deg if that puts the narrower
        # side between the fingers (the 2F-85 has little clearance on a soup can).
        perp = np.array([-axis[1], axis[0], 0.0])
        w0, w90 = abs(size[:2] @ axis[:2]), abs(size[:2] @ perp[:2])
        self.yaw = 0.0
        if min(w0, w90) <= self.SMALL_OBJECT and w90 < w0 - 0.003:
            self.yaw, axis = np.pi / 2, perp
        width_along_axis = abs(size[:2] @ axis[:2])
        pad_drop = self._pad_offset()
        if width_along_axis <= self.SMALL_OBJECT:
            grasp_xy = ocen[:2]
            # put the finger pads (not the TCP site) low on the object
            grasp_z = max(olo[2] + 0.012, ohi[2] - self.CENTER_DEPTH) - pad_drop
        else:  # rim grasp: straddle the wall nearest the robot along the closing axis
            half = width_along_axis / 2
            grasp_xy = ocen[:2] + axis[:2] * (half - self.RIM_INSET)
            grasp_z = ohi[2] - min(self.RIM_DEPTH, 0.85 * size[2])
        grasp_offset_xy = grasp_xy - ocen[:2]  # keep the same object-relative grasp
        hold_height = grasp_z - olo[2]  # TCP height above the object's bottom
        tcen = (tlo + thi) / 2
        place_xy = tcen[:2] + grasp_offset_xy
        place_z = thi[2] + hold_height + 0.015
        travel_z = max(ohi[2], thi[2]) + size[2] + 0.12
        g_open, g_close = -1.0, 1.0
        self.waypoints = [
            # (xyz target, gripper, min_steps, tolerance, max speed in action units)
            (np.r_[grasp_xy, travel_z], g_open, 0, 0.02, 1.0),
            (np.r_[grasp_xy, grasp_z + 0.06], g_open, 0, 0.005, 1.0),
            (np.r_[grasp_xy, grasp_z], g_open, 0, 0.006, 0.35),  # slow final approach
            (np.r_[grasp_xy, grasp_z], g_close, self.CLOSE_STEPS, 1.0, 1.0),
            (np.r_[grasp_xy, travel_z], g_close, 0, 0.02, 0.8),
            (np.r_[place_xy, travel_z], g_close, 0, 0.015, 0.8),
            (np.r_[place_xy, place_z], g_close, 0, 0.008, 0.4),
            (np.r_[place_xy, place_z], g_open, 15, 1.0, 1.0),
            (np.r_[place_xy, travel_z], g_open, 0, 0.03, 1.0),
        ]
        if self.verbose:
            print(f"pick {obj} (w={width_along_axis:.3f}) -> {target}; grasp z={grasp_z:.3f}")

    def reset(self, prompt):
        self.home_rot = None
        self.yaw = 0.0
        self.waypoints = None
        self.idx = 0
        self.k = 0
        self.trace = []

    def act(self, obs):
        raw = self.env.raw_obs
        pos = raw["robot0_eef_pos"]
        rot = T.quat2mat(raw["robot0_eef_quat"])
        if self.waypoints is None:
            self._plan()
            c, sn = np.cos(self.yaw), np.sin(self.yaw)
            self.home_rot = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ rot
        target, grip, min_steps, tol, speed = self.waypoints[min(self.idx, len(self.waypoints) - 1)]
        err = target - pos
        self.k += 1
        self.trace.append(pos.copy())
        reached = np.linalg.norm(err) < tol
        if self.idx == 1:  # don't start the final approach until the fingers are fully open
            reached = reached and self.env.gripper_open_fraction() > 0.97
        # blocked (e.g. object resting on a rim before the exact place height): move on
        stalled = self.idx != 1 and tol < 1 and self.k > 20 and np.linalg.norm(self.trace[-1] - self.trace[-15]) < 0.002
        if self.idx < len(self.waypoints) and ((self.k >= min_steps and reached) or stalled or self.k > 120):
            self.idx += 1
            self.k = 0
        dpos = np.clip(err * self.POS_GAIN, -speed, speed)
        r_err = T.quat2axisangle(T.mat2quat(self.home_rot @ rot.T))
        drot = np.clip(r_err * self.ROT_GAIN, -1, 1)
        return np.r_[dpos, drot, grip]


# --------------------------------------------------------------------------- #
# Remote policy over websocket (openpi-compatible wire format)                 #
# --------------------------------------------------------------------------- #
from .wire import pack as _pack, unpack as _unpack  # noqa: E402


class RemotePolicy(Policy):
    """Talks to a policy server over websocket using openpi's msgpack-numpy format.

    key_format="openpi" sends the keys openpi's LIBERO policies expect
    (observation/image, observation/wrist_image, observation/state, prompt), so
    you can point this at ``scripts/serve_policy.py`` in openpi as-is.
    key_format="raw" sends this framework's observation dict unchanged
    (see libero6dof.server for a matching server).
    """

    def __init__(self, host="localhost", port=8000, key_format="openpi", image_size=224):
        from websockets.sync.client import connect

        self.url = f"ws://{host}:{port}"
        self.key_format = key_format
        self.image_size = image_size
        self.ws = connect(self.url, compression=None, max_size=None, open_timeout=60)
        self.metadata = _unpack(self.ws.recv())

    def _resize(self, img):
        if img.shape[0] == self.image_size:
            return img
        import cv2

        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def act(self, obs):
        if self.key_format == "openpi":
            msg = {
                "observation/image": self._resize(obs["image"]),
                "observation/wrist_image": self._resize(obs["wrist_image"]),
                "observation/state": obs["state"],
                "prompt": obs["prompt"],
            }
        else:
            msg = {k: v for k, v in obs.items()}
        self.ws.send(_pack(msg))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"policy server error:\n{resp}")
        out = _unpack(resp)
        return np.asarray(out["actions"], dtype=np.float32)

    def close(self):
        self.ws.close()
