# pi0.5 base as a LIBERO policy server (run in env_pi). Zero-shot: no LIBERO training. The base checkpoint ships
# without normalization stats, so state/actions are mapped with the LIBERO dataset's min/max range.
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from policy_io import serve
from pi05_common import load_pi05, to_model_image

# physical-intelligence/libero meta/stats.json (min/max used in place of q01/q99, which the dataset doesn't ship)
STATE_MIN = np.array([-0.483, -0.326, 0.008, 0.353, -3.641, -1.843, -0.001, -0.042])
STATE_MAX = np.array([0.21, 0.391, 1.366, 3.671, 3.561, 1.386, 0.042, 0.001])
ACT_MIN = np.array([-0.938, -0.938, -0.938, -0.258, -0.375, -0.368, -1.0])
ACT_MAX = np.array([0.938, 0.938, 0.938, 0.356, 0.375, 0.375, 1.0])
REPLAN = 5                       # run 5 of the 50 predicted actions, then predict again (openpi's LIBERO setting)

class Pi05Base:
    def __init__(self):
        self.policy, self.tok = load_pi05()
        self.model = self.policy.model

    def reset(self, instruction):
        self.instruction, self.queue = instruction, []
        return True

    @torch.no_grad()
    def act(self, obs):
        if not self.queue:
            state = 2 * (obs["state"] - STATE_MIN) / (STATE_MAX - STATE_MIN) - 1
            state = np.pad(np.clip(state, -1, 1), (0, 32 - len(state)))           # pad to 32 like openpi
            bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            prompt = f"Task: {self.instruction.strip().lower()}, State: {' '.join(map(str, bins))};\nAction: "
            enc = self.tok(prompt, padding="max_length", max_length=200, truncation=True, return_tensors="pt")
            tokens, mask = enc["input_ids"].to("cuda"), enc["attention_mask"].bool().to("cuda")
            images = [to_model_image(obs["front"]), to_model_image(obs["wrist"]), torch.full((1, 3, 224, 224), -1.0, device="cuda")]
            img_masks = [torch.ones(1, dtype=torch.bool, device="cuda")] * 2 + [torch.zeros(1, dtype=torch.bool, device="cuda")]
            chunk = self.model.sample_actions(images, img_masks, tokens, mask)[0, :REPLAN, :7].float().cpu().numpy()
            acts = (np.clip(chunk, -1, 1) + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN   # LIBERO units, gripper -1 open / +1 close
            acts[:, -1] = np.sign(acts[:, -1])
            self.queue = list(acts)
        return self.queue.pop(0)

if __name__ == "__main__":
    serve(Pi05Base(), int(os.environ.get("PORT", 5555)))
