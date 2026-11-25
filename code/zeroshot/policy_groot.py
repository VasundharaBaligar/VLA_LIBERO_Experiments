# GR00T N1.5 base as a LIBERO policy server (run in env_groot). Zero-shot: no LIBERO training.
# LIBERO's robot is a single Franka arm with end-effector delta control, so we drive the base model through its
# pretraining embodiment closest to that: DROID ("oxe_droid", Franka, eef deltas in [-1, 1], gripper 0=open 1=closed).
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from policy_io import serve
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

REPLAN = 8                       # run 8 of the 16 predicted actions, then predict again

class GrootBase:
    def __init__(self):
        cfg = DATA_CONFIG_MAP["oxe_droid"]
        self.policy = Gr00tPolicy(model_path=os.environ.get("GROOT_MODEL", "nvidia/GR00T-N1.5-3B"),
                                  embodiment_tag="oxe_droid", modality_config=cfg.modality_config(),
                                  modality_transform=cfg.transform(), denoising_steps=4)

    def reset(self, instruction):
        self.instruction, self.queue = instruction, []
        return True

    def act(self, obs):
        if not self.queue:
            quat = Rotation.from_rotvec(obs["state"][3:6])           # client sends axis-angle
            gripper = 1 - np.clip(obs["state"][6] / 0.04, 0, 1)        # LIBERO finger qpos 0.04 open -> DROID 0 open
            out = self.policy.get_action({
                "video.exterior_image_1": obs["front"][None],
                "video.exterior_image_2": obs["front"][None],         # DROID has 2 side cameras; LIBERO has one
                "video.wrist_image": obs["wrist"][None],
                "state.eef_position": obs["state"][None, :3].astype(np.float64),
                "state.eef_rotation": quat.as_euler("xyz")[None],
                "state.gripper_position": np.array([[gripper]]),
                "annotation.language.language_instruction": [self.instruction.lower()],
            })
            acts = np.concatenate([out["action.eef_position_delta"], out["action.eef_rotation_delta"],
                                   np.where(out["action.gripper_position"] > 0.5, 1.0, -1.0)], axis=-1)[:REPLAN]
            self.queue = list(np.clip(acts, -1, 1))
        return self.queue.pop(0)

if __name__ == "__main__":
    serve(GrootBase(), int(os.environ.get("PORT", 5556)))
