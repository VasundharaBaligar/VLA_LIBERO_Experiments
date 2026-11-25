# One figure per scene: every object (rows) x [true object, VLM, VLA base, VLA fine-tuned on that suite (if run)].
# Usage: python compare_heatmaps.py openvla | pi05
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from heatmap_common import load_scenes, score, upsample

DIR = "/workspace/fa26-vla-cs670/zeroshot_results/heatmaps"
MODEL = sys.argv[1]
OUT = f"{DIR}/compare/{MODEL}"
os.makedirs(OUT, exist_ok=True)
NAMES = {"openvla": ("Prismatic", "OpenVLA"), "pi05": ("PaliGemma", "pi0.5"),
         "groot": ("Eagle2-2B stand-in", "GR00T N1.5")}[MODEL]
STAGES = [("vlm", f"VLM ({NAMES[0]})\nno robot training"), ("vla", f"{NAMES[1]} base\nrobot pretraining"),
          ("ft", f"{NAMES[1]} fine-tuned\non this suite")]
grids = {s: np.load(f"{DIR}/{MODEL}_{s}_grids.npz") for s in ["vlm", "vla"]}

for scene in load_scenes():
    ft_path = f"{DIR}/{MODEL}_ft_{scene['suite']}_grids.npz"
    ft = np.load(ft_path) if os.path.exists(ft_path) else None
    stages = STAGES if ft is not None else STAGES[:2]
    objs = scene["objects"]
    fig, axes = plt.subplots(len(objs), len(stages) + 1, figsize=(2.75 * (len(stages) + 1), 2.9 * len(objs)))
    for i, o in enumerate(objs):
        k = f"{scene['key']}|{o['name']}"
        ax = axes[i, 0]; ax.imshow(scene["img"]); ax.contour(o["mask_arr"], levels=[0.5], colors="lime", linewidths=1.5)
        ax.set_title(f"\"{o['name']}\"", fontsize=11, fontweight="bold"); ax.axis("off")
        for j, (s, name) in enumerate(stages):
            g = (ft if s == "ft" else grids[s])[k]
            hit, lift, (px, py) = score(g, o["mask_arr"])
            ax = axes[i, j + 1]; ax.imshow(scene["img"]); ax.imshow(upsample(g), cmap="jet", alpha=0.5)
            ax.contour(o["mask_arr"], levels=[0.5], colors="lime", linewidths=0.8)
            ph = 256 // g.shape[0]
            ax.add_patch(plt.Rectangle((px * ph, py * ph), ph, ph, fill=False, ec="white", lw=2))
            ax.set_title(("" if i else name + "\n") + f"{'HIT' if hit else 'miss'} | lift {lift:.1f}x",
                         fontsize=9, color="green" if hit else "red"); ax.axis("off")
    fig.suptitle(f"{scene['suite']}: \"Where is the <object>?\"  (green = true object, white box = hottest spot)",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/{scene['key']}.png", dpi=100); plt.close(fig)
    print("saved", f"{OUT}/{scene['key']}.png")
