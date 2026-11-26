"""Stage 2 summary of dry-run episode files: sizes, a few rows, label distributions, Stage 3 disk/time estimate.
Usage: python checks/stage2_summary.py data/dryrun/libero_spatial/*.pt"""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from labels import DIRECTIONS, direction_label   # noqa: E402

paths = sorted(p for p in sys.argv[1:] if p.endswith(".pt"))
eps = [torch.load(p, weights_only=True) for p in paths]

print("== files")
for p, e in zip(paths, eps):
    print(f"  {os.path.basename(p)}: {os.path.getsize(p) / 1e6:.2f} MB | queries {e['n_queries']} | "
          f"sim steps {len(e['trace']['t'])} | success {e['success']} | {e['seconds']:.0f} s | error {e['error']}")
    print(f"     hidden {tuple(e['hidden'].shape)} {e['hidden'].dtype} = (queries, positions {e['hidden_positions']}, "
          f"points {e['hidden_points']}, 4096)")

e = eps[0]
print("\n== a few saved rows (episode", e["episode_id"] + ")")
for q in [0, 1, e["n_queries"] // 2, e["n_queries"] - 1]:
    print(f"  q{q:02d} t={int(e['t'][q]):3d} next_xyz={np.round(e['next_xyz'][q].numpy(), 3)} "
          f"close_next={bool(e['gripper_close_next'][q])} contact={bool(e['contact'][q])} "
          f"phase={'after' if e['phase_after'][q] else 'before'} bowl_lift={float(e['bowl_lift'][q]) * 100:.1f}cm "
          f"bowl_xy={np.round(e['bowl_xy'][q].numpy(), 1)} plate_xy={np.round(e['plate_xy'][q].numpy(), 1)} "
          f"target_xy={np.round(e['target_xy'][q].numpy(), 1)} | hidden[act7,L16,:4]="
          f"{e['hidden'][q, 1, 4, :4].float().numpy().round(3)}")

xyz = torch.cat([e["next_xyz"] for e in eps]).numpy()
n = len(xyz)
print(f"\n== label distributions over {n} queries ({len(eps)} episodes)")
mag = np.abs(xyz).max(1)
print("  max |xyz component| quantiles (5,10,25,50,75,90%):", np.round(np.quantile(mag, [.05, .1, .25, .5, .75, .9]), 4))
for thr in (0.02, 0.05, 0.1):
    d = np.array([direction_label(v, thr) for v in xyz])
    print(f"  direction with stay threshold {thr}: " + ", ".join(f"{DIRECTIONS[k]} {np.mean(d == k):.0%}"
                                                              for k in range(7)))
dist = np.linalg.norm(xyz, axis=1)
print("  distance ||xyz|| 8-bin quantile edges (provisional, from these episodes):",
      np.round(np.quantile(dist, np.linspace(0, 1, 9)[1:-1]), 3))
for k in ("gripper_close_next", "contact", "phase_after", "grasped_now"):
    v = torch.cat([e[k] for e in eps]).numpy()
    print(f"  {k}: {v.mean():.0%} true")
txy = torch.cat([e["target_xy"] for e in eps]).numpy()
print(f"  target_xy: x {txy[:, 0].min():.0f}-{txy[:, 0].max():.0f}, y {txy[:, 1].min():.0f}-{txy[:, 1].max():.0f} px "
      f"(256x256 frame)")

mb_q = np.mean([os.path.getsize(p) / 1e6 / e["n_queries"] for p, e in zip(paths, eps)])
s_q = np.mean([e["seconds"] / e["n_queries"] for e in eps])
print(f"\n== Stage 3 estimate: {mb_q * 1000:.0f} KB and {s_q:.2f} s per query (s/query includes the sim steps; "
      f"with --check the model runs twice per query)")
for name, q in [("LIBERO-Spatial 500 eps (~16 queries each if ~97% succeed)", 500 * 16),
                ("LIBERO-PRO swap 500 eps (up to 28 queries each if they fail)", 500 * 28)]:
    print(f"  {name}: {q} queries -> {q * mb_q / 1000:.1f} GB")
