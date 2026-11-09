# libero6dof

A LIBERO simulation framework for **6-DOF robot arms**: run every LIBERO task (spatial, object, goal, long-horizon, LIBERO-90) with a UR5e (or your own 6-DOF arm), generate demonstrations for it, and evaluate VLA policies served over a websocket.

Stock LIBERO is hard-wired to the 7-DOF Franka Panda. Its benchmark init states and demo datasets are Panda-only. This package handles the parts that break when you swap the arm:

| Problem with stock LIBERO | What libero6dof does |
|---|---|
| Only `MountedPanda` / `OnTheGroundPanda` robot classes | `robots.py` generates `Mounted<Arm>` / `OnTheGround<Arm>` for any 6-DOF `ArmSpec`, so every BDDL task loads with `robots=["UR5e"]` |
| The arm must start where LIBERO expects | `calibrate.py` runs IK so the 6-DOF TCP starts at the Panda's exact home pose (UR5e: 0.02 mm, 0.000° error) |
| Fixed eval layouts (`*.pruned_init`) are Panda MuJoCo states | `tasks.transfer_init_states` copies every object / drawer / door joint into the 6-DOF scene. Same 50 layouts per task, cached in `~/.libero6dof` |
| robosuite 1.4's Robotiq 2F-85 jams half-closed on MuJoCo 3 (finger self-collision) | `grippers.Robotiq85GripperFixed` adds contact exclusions, so it closes and grips properly |
| LIBERO demos are Panda-only | `collect.py` generates successful demos for your arm with a scripted expert, in LIBERO's HDF5 layout |
| LIBERO's init-state loader breaks on torch ≥ 2.6 | Loader patched in `tasks.panda_init_states` |

## Install (Ubuntu 22.04 / 24.04)

```bash
bash scripts/install.sh          # uv + Python 3.10 venv, LIBERO clone, LIBERO config
source .venv/bin/activate
bash scripts/smoke_test.sh       # ~5 min: scripted expert + videos + policy-server round trip
```

Rendering is headless via EGL by default. Use `MUJOCO_GL=osmesa` on a machine without a GPU, or `MUJOCO_GL=glfw` for an on-screen window.

## Interfaces

**Observation** (`Libero6DofEnv.reset()` / `.step()`):

| key | shape | notes |
|---|---|---|
| `image` | (H,W,3) uint8 | agentview camera, rotated 180° like openpi / OpenVLA LIBERO data |
| `wrist_image` | (H,W,3) uint8 | eye-in-hand camera |
| `state` | (8,) | `[eef_pos(3), eef_axis_angle(3), finger_l, finger_r]`. Same layout as openpi's LIBERO state. The 6-DOF gripper opening is mapped to Panda finger widths (±0.04 open, 0 closed) |
| `joint_pos` | (6,) | raw arm joints |
| `gripper_open` | float | 1 open → 0 closed |
| `prompt` | str | language instruction |

**Action**: `(7,)` in [-1, 1]: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. These are OSC_POSE end-effector deltas in the world frame, exactly as in stock LIBERO; gripper −1 opens, +1 closes. Policies can also return an `(H, 7)` chunk, which is executed for `--replan-steps` steps before re-querying.

Because control is in end-effector space, **the action space doesn't change between a 7-DOF and a 6-DOF arm**. Only reachability, wrist dynamics and appearance differ.

## Usage

```python
from libero6dof.env import Libero6DofEnv
env = Libero6DofEnv("libero_spatial", task_id=0, robot="UR5e", image_size=256)
obs = env.reset(episode_idx=0)            # one of LIBERO's 50 fixed layouts
obs, reward, done, info = env.step(action) # info["success"]
```

```bash
# Scripted expert (reachability check / baseline)
python -m libero6dof.evaluate --suite libero_spatial --policy scripted --episodes 10 --video-dir runs/vid

# Generate fine-tuning data for your arm
python -m libero6dof.collect --suite libero_spatial --demos 50 --out data/ur5e

# Evaluate a VLA
python -m libero6dof.evaluate --suite libero_spatial --policy remote --host localhost --port 8000
```

### Plugging in a VLA

Run the model in its own environment (its own Python/CUDA, even another machine) and serve it over a websocket:

* **openpi (π0 / π0.5)**: start `scripts/serve_policy.py` in openpi as usual. `--policy remote` uses openpi's msgpack-numpy wire format and LIBERO keys (`observation/image`, `observation/wrist_image`, `observation/state`, `prompt`) by default.
* **Anything else** (SmolVLA / LeRobot, GR00T, OpenVLA-OFT, your own model): write a factory that returns an object with `act(obs) -> (7,) or (H,7)`, then run
  `python -m libero6dof.server --policy my_pkg.my_module:make_policy --port 8000`.

A checkpoint fine-tuned on Panda LIBERO data will **not** transfer zero-shot to the UR5e. The wrist camera view, arm appearance and grasp geometry all differ. Expect to fine-tune on 6-DOF demos from `collect.py`.

## Training SmolVLA (molab GPU) and evaluating locally

Two environments, because LIBERO pins old NumPy and robosuite:

| env | install | used for |
|---|---|---|
| `.venv` (Python 3.10) | `bash scripts/install.sh` | simulator: collect demos, evaluate |
| `.venv-lerobot` (Python 3.12) | `bash scripts/install_lerobot.sh` | convert to LeRobot, serve checkpoints |

**1. Collect demos** (sim env). Use 256 px images, which matches LIBERO datasets:
```bash
python -m libero6dof.collect --suite libero_spatial --demos 50 --image-size 256 --out data/ur5e
```

**2. Convert and push to the Hub** (lerobot env, after `hf auth login`):
```bash
python -m libero6dof.to_lerobot --src data/ur5e --repo-id <you>/libero6dof_ur5e_spatial --push
```
Keys follow LeRobot's LIBERO convention (`observation.images.image`, `observation.images.image2`, 8-D `observation.state`, 7-D `action`, 20 fps).

**3. Train on molab.** Put this repo on GitHub, then in molab use the new-notebook dropdown and paste the URL of `notebooks/train_smolvla_molab.py`. Then:
- attach the GPU (notebook specs button)
- add an `HF_TOKEN` secret with write access
- fill in the form
- press **Dry run**, then **Start training**

Checkpoints, including optimizer state, are pushed to your model repo every `save_freq` steps (`--save_checkpoint_to_hub`). When a session hits molab's 12-hour cap or the 90-minute idle shutdown, reopen the notebook, pick **Resume**, and train again from the latest checkpoint. Keep the tab open during long runs; it's unclear whether a running cell counts as activity for the idle timer.

The notebook maps your two cameras onto the camera names `smolvla_base` was pretrained with (`--rename_map`). The mapping is stored in the checkpoint, and the serving adapter reads it back.

**4. Evaluate locally:**
```bash
# lerobot env: serve the checkpoint (Hub repo id or local folder)
LEROBOT_CHECKPOINT=<you>/smolvla_ur5e_spatial \
  python -m libero6dof.server --policy libero6dof.lerobot_policy:make_policy --port 8000
# sim env
python -m libero6dof.evaluate --suite libero_spatial --policy remote --port 8000 --replan-steps 10 --video-dir runs/smolvla
```
`lerobot_policy.py` works for any LeRobot checkpoint (SmolVLA, pi0, pi0.5, ACT, ...). It resizes images to the training resolution and returns the full action chunk; `--replan-steps` sets how much of each chunk is executed.

### Adding your own 6-DOF arm

See `configs/custom_arm.yaml`. In short: make a robosuite-style MJCF, fill in the YAML, then run
`python -m libero6dof.calibrate --arm-config configs/custom_arm.yaml --write`, and pass `--arm-config` to `evaluate` / `collect`.

## Demo HDF5 layout

One file per task, `data/<suite>/taskNN_<name>_demo.hdf5`. It mirrors LIBERO's layout (`data/demo_i/{actions, states, rewards, dones, obs/{agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, ee_states, gripper_states, joint_states}}`). Images are stored already rotated (`data.attrs["image_flip"]`), so don't flip them again in a converter written for raw LIBERO data.

## Scripted expert coverage

`ScriptedPickPlace` handles single "pick X → place on/in Y" goals. It reads object poses from the simulator, so treat it as a **data generator and reachability check, not a baseline**. It skips goals it can't do: opening drawers, turning on the stove, multi-object goals.

Measured with the UR5e + Robotiq 2F-85 in this framework, using LIBERO's fixed layouts and horizons:

| suite | success | notes |
|---|---|---|
| libero_spatial | 46% (50 eps) | 100% on 2 tasks. 0% when the bowl is in or on the cabinet |
| libero_object | 18% (50 eps) | Tall, light bottles and cans tip over during the approach. The 2F-85 has only a few mm of clearance on them |
| libero_goal | 17% (24 eps) | 2 of 10 goals unsupported |

`collect.py` keeps only successful rollouts and retries with action noise, so a low rate means more attempts per demo (`--max-attempts`), not bad data. For tasks it can't do, record demos by teleoperation (robosuite's keyboard / SpaceMouse devices work with `Libero6DofEnv.env`) or train on the tasks it covers first. Grasp constants are class attributes on `ScriptedPickPlace` (`RIM_INSET`, `RIM_DEPTH`, `CENTER_DEPTH`, `SMALL_OBJECT`) if you want to tune it for another gripper.

## Layout

```
libero6dof/
  robots.py      ArmSpec + registration of Mounted/OnTheGround classes
  grippers.py    Robotiq 2F-85 self-collision fix
  calibrate.py   IK to match LIBERO's home end-effector pose
  tasks.py       suites, env construction, init-state transfer
  env.py         Libero6DofEnv (VLA-ready obs/action)
  policies.py    Policy API, ScriptedPickPlace, RemotePolicy (websocket client)
  server.py      websocket policy server
  evaluate.py    suite evaluation CLI (+ videos, results.json)
  collect.py     demo generation CLI (HDF5)
  to_lerobot.py  HDF5 demos -> LeRobot dataset (+ Hub push)
  lerobot_policy.py  serve any LeRobot checkpoint to the simulator
  wire.py        openpi-compatible msgpack-numpy format (no sim imports)
notebooks/train_smolvla_molab.py   marimo notebook for molab GPU training
configs/custom_arm.yaml
scripts/install.sh, scripts/install_lerobot.sh, scripts/smoke_test.sh
```
