import numpy as np


class MyVLA:
    def __init__(self, checkpoint: str):
        # Load your model here, e.g. a LeRobot SmolVLA / pi0 checkpoint fine-tuned on
        # libero6dof demos, a GR00T checkpoint, or an OpenVLA-OFT model.
        self.checkpoint = checkpoint

    def act(self, obs: dict) -> np.ndarray:
        image = obs["observation/image"]
        wrist = obs["observation/wrist_image"]
        state = obs["observation/state"]
        prompt = obs["prompt"]
        # actions = self.model.predict(image, wrist, state, prompt)
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 6] = -1.0  # placeholder: hold still, gripper open
        return actions


def make_policy():
    return MyVLA(checkpoint="path/to/checkpoint")
