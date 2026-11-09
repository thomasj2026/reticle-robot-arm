from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm
from robosuite.utils.mjcf_utils import xml_path_completion

from . import grippers  # noqa: F401  (registers Robotiq85GripperFixed)

# Workspace offsets copied from LIBERO's MountedPanda / OnTheGroundPanda so the
# arm base sits where LIBERO's scenes expect it.
_MOUNTED_OFFSETS = {
    "bins": (-0.5, -0.1, 0),
    "empty": (-0.6, 0, 0),
    "table": lambda L: (-0.16 - L / 2, 0, 0),
    "study_table": lambda L: (-0.25 - L / 2, 0, 0),
    "kitchen_table": lambda L: (-0.16 - L / 2, 0, 0),
}
_GROUND_OFFSETS = {
    "bins": (-0.5, -0.1, 0),
    "empty": (-0.6, 0, 0),
    "table": lambda L: (-0.16 - L / 2, 0, 0),
    "coffee_table": lambda L: (-0.16 - L / 2, 0, 0.41),
    "living_room_table": lambda L: (-0.16 - L / 2, 0, 0.42),
}


@dataclass
class ArmSpec:
    """Everything needed to drop a 6-DOF arm into LIBERO.

    name:        Name used in ``robots=[name]`` (e.g. "UR5e").
    mjcf:        robosuite asset path ("robots/ur5e/robot.xml") or an absolute
                 path to your own MJCF. The MJCF must follow robosuite's
                 conventions (joints named robot0_*, a "right_hand" body, etc.).
    gripper:     robosuite gripper name ("Robotiq85Gripper", "PandaGripper", ...).
    home_qpos:   Joint angles at reset. Use ``python -m libero6dof.calibrate``
                 to compute one that matches LIBERO's Panda end-effector pose.
    mount:       robosuite mount used in table scenes (floor scenes use none,
                 matching LIBERO's OnTheGroundPanda).
    joint_damping: Optional per-joint damping (len == dof).
    """

    name: str
    mjcf: str
    gripper: str
    home_qpos: Sequence[float]
    mount: str | None = "RethinkMount"
    joint_damping: Sequence[float] | None = None
    extra_offsets: dict[str, object] = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return len(self.home_qpos)


# Home pose solved by `python -m libero6dof.calibrate --arm UR5e` so the
# Robotiq TCP starts at LIBERO's Panda home pose, pointing straight down.
UR5E = ArmSpec(
    name="UR5e",
    mjcf="robots/ur5e/robot.xml",
    gripper="Robotiq85GripperFixed",
    home_qpos=[-0.29212, -1.66180, 1.85932, -1.71392, -1.58715, -1.86296],
)

ARM_SPECS: dict[str, ArmSpec] = {"UR5e": UR5E}
_REGISTERED: dict[str, ArmSpec] = {}


def _make_model_class(spec: ArmSpec, prefix: str, offsets: dict, mount):
    xml = spec.mjcf if os.path.isabs(spec.mjcf) else xml_path_completion(spec.mjcf)
    home = np.array(spec.home_qpos, dtype=float)

    def __init__(self, idn=0):
        ManipulatorModel.__init__(self, xml, idn=idn)
        if spec.joint_damping is not None:
            self.set_joint_attribute(attrib="damping", values=np.array(spec.joint_damping))

    attrs = {
        "__init__": __init__,
        "__doc__": f"{spec.name} ({spec.dof}-DOF) placed for LIBERO {prefix} scenes.",
        "default_mount": property(lambda self: mount),
        "default_gripper": property(lambda self: spec.gripper),
        "default_controller_config": property(lambda self: "default_ur5e"),
        "init_qpos": property(lambda self: home.copy()),
        "base_xpos_offset": property(lambda self: {**offsets, **spec.extra_offsets}),
        "top_offset": property(lambda self: np.array((0, 0, 1.0))),
        "_horizontal_radius": property(lambda self: 0.5),
        "arm_type": property(lambda self: "single"),
    }
    # type(ManipulatorModel) is robosuite's RobotModelMeta, which auto-registers.
    return type(ManipulatorModel)(f"{prefix}{spec.name}", (ManipulatorModel,), attrs)


def register_arm(spec: ArmSpec) -> ArmSpec:
    """Register (or re-register with a new home pose) a 6-DOF arm for LIBERO."""
    # Like LIBERO's Panda: pedestal mount for table scenes, base on the floor otherwise.
    for prefix, offsets, mount in (
        ("Mounted", _MOUNTED_OFFSETS, spec.mount),
        ("OnTheGround", _GROUND_OFFSETS, None),
    ):
        cls = _make_model_class(spec, prefix, offsets, mount)
        REGISTERED_ROBOTS[cls.__name__] = cls
        ROBOT_CLASS_MAPPING[cls.__name__] = SingleArm
    ARM_SPECS[spec.name] = spec
    _REGISTERED[spec.name] = spec
    return spec


def get_arm(name: str) -> ArmSpec:
    if name not in _REGISTERED:
        if name not in ARM_SPECS:
            raise KeyError(f"Unknown arm '{name}'. Known: {list(ARM_SPECS)}")
        register_arm(ARM_SPECS[name])
    return _REGISTERED[name]