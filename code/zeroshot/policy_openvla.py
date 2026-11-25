# OpenVLA-7B base (no LIBERO fine-tuning), the starting point of OpenVLA-OFT.
# The base model needs an un-normalization key from one of its pretraining datasets; we use a
# Franka dataset whose actions are robosuite-style end-effector deltas, the closest match to LIBERO.
import os
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

UNNORM_KEY = os.environ.get("OPENVLA_UNNORM_KEY", "viola")

class OpenVLABase:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
        assert UNNORM_KEY in self.vla.norm_stats, sorted(self.vla.norm_stats)

    def reset(self, instruction):
        self.prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        return True

    def act(self, obs):
        img = Image.fromarray(obs["front"]).resize((224, 224), Image.LANCZOS)   # no crop: base had no crop augmentation
        inputs = self.processor(self.prompt, img).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            a = np.asarray(self.vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False), dtype=np.float64)
        a[-1] = -np.sign(2 * a[-1] - 1)          # OpenVLA gripper 1=open -> LIBERO -1=open
        return a
