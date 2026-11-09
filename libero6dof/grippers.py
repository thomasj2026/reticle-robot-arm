from __future__ import annotations

import itertools

from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.robotiq_85_gripper import Robotiq85Gripper
from robosuite.utils.mjcf_utils import new_element


def _exclude_internal_contacts(gripper):
    bodies = [b.get("name") for b in gripper.worldbody.iter("body") if b.get("name")]
    for b1, b2 in itertools.combinations(bodies, 2):
        gripper.contact.append(new_element("exclude", name=None, body1=b1, body2=b2))
    return gripper


class Robotiq85GripperFixed(Robotiq85Gripper):
    """Robotiq 2F-85 without the self-collision that stalls it half-closed."""

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        _exclude_internal_contacts(self)


GRIPPER_MAPPING["Robotiq85GripperFixed"] = Robotiq85GripperFixed
