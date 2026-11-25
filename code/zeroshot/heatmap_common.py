# Shared scoring + plotting for object-location heatmaps (used by every model's heatmap script).
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCENES = "/workspace/fa26-vla-cs670/zeroshot_results/scenes"

def load_scenes():
    meta = json.load(open(f"{SCENES}/scenes.json"))
    for e in meta:
        e["img"] = np.array(Image.open(f"{SCENES}/{e['image']}"))
        for o in e["objects"]:
            o["mask_arr"] = np.load(f"{SCENES}/{o['mask']}")
    return meta

def upsample(grid, size=256):
    g = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0

def score(grid, mask):
    """grid: (h, w) attention over image patches. Hit = hottest patch overlaps the object mask.
    Lift = share of attention on the object / share of image the object covers (1.0 = chance)."""
    h, w = grid.shape
    py, px = np.unravel_index(np.argmax(grid), grid.shape)
    ph, pw = mask.shape[0] // h, mask.shape[1] // w
    hit = bool(mask[py * ph:(py + 1) * ph, px * pw:(px + 1) * pw].any())
    p = grid / grid.sum()
    patch_cover = mask.reshape(h, ph, w, pw).mean(axis=(1, 3))      # fraction of each patch on the object
    lift = float((p * patch_cover).sum() / patch_cover.mean())
    return hit, lift, (px, py)

def plot(rows, title, out_path):
    """rows: list of dicts with img, mask, grid, label, hit, lift, peak."""
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7))
    for i, r in enumerate(rows):
        ax = axes[0, i]; ax.imshow(r["img"]); ax.contour(r["mask"], levels=[0.5], colors="lime", linewidths=1.5)
        if r.get("box"):                                   # model's own "detect" answer, if it gives one
            x0, y0, x1, y1 = r["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="magenta", lw=2))
        ax.set_title(f"{r['scene']}\n\"{r['label']}\" (true object in green"
                     f"{', detect box in magenta' if 'box' in r else ''})", fontsize=9); ax.axis("off")
        ax = axes[1, i]; ax.imshow(r["img"]); ax.imshow(upsample(r["grid"]), cmap="jet", alpha=0.5)
        ph = 256 // r["grid"].shape[0]; px, py = r["peak"]
        ax.add_patch(plt.Rectangle((px * ph, py * ph), ph, ph, fill=False, ec="white", lw=2))
        ax.set_title(f"{'HIT' if r['hit'] else 'miss'}  |  lift {r['lift']:.2f}x", fontsize=10,
                     color="green" if r["hit"] else "red"); ax.axis("off")
    fig.suptitle(title, fontsize=12); plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close(fig)

def save_grids(rows, model, stage, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{model}_{stage}_grids.npz", **{f"{r['key']}|{r['label']}": r["grid"] for r in rows})

def save_summary(rows, model, stage, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summ = [dict(model=model, stage=stage, scene=r["scene"], object=r["label"], hit=r["hit"], lift=round(r["lift"], 2))
            for r in rows]
    json.dump(summ, open(f"{out_dir}/{model}_{stage}_heatmap_scores.json", "w"), indent=1)
    print(f"{model} {stage}: {sum(r['hit'] for r in rows)}/{len(rows)} hits, mean lift "
          f"{np.mean([r['lift'] for r in rows]):.2f}x")
    for s in summ:
        print("   ", s)
