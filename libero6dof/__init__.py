"""libero6dof: LIBERO simulation framework for 6-DOF robot arms."""

import os

# Headless rendering by default. Override with MUJOCO_GL=glfw for an on-screen viewer
# or MUJOCO_GL=osmesa on machines without a GPU / EGL.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

__version__ = "0.1.0"
