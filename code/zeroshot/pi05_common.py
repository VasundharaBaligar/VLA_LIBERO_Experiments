# Shared loading for pi0.5 base (no LIBERO fine-tuning) via LeRobot, used by the policy server and heatmaps.
import numpy as np, torch
from transformers import AutoTokenizer
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

REPO = "lerobot/pi05_base"

def load_pi05():
    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device, cfg.dtype = "cuda", "bfloat16"
    policy = PI05Policy.from_pretrained(REPO, config=cfg).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    return policy, tok

def to_model_image(img_uint8):
    """HxWx3 uint8 -> (1, 3, 224, 224) float in [-1, 1], as pi0.5 / SigLIP expect."""
    t = torch.from_numpy(img_uint8).float().div(255).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return (t * 2 - 1).to("cuda")
