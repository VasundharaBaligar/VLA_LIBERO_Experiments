"""Success rate per task and overall from collected episode files (reads metadata only; hidden states are skipped).
Usage: python checks/success_rates.py data/collected/libero_spatial"""
import glob, os, sys
from collections import defaultdict

import torch

d = sys.argv[1]
by = defaultdict(list)
errors = []
for p in sorted(glob.glob(os.path.join(d, "task*_ep*.pt"))):
    e = torch.load(p, weights_only=True, mmap=True)
    by[int(e["task_id"])].append(bool(e["success"]))
    if e["error"]:
        errors.append((os.path.basename(p), e["error"]))
    task_names = by.setdefault("names", {})
    task_names[int(e["task_id"])] = (e["task"], e.get("wording_mismatch"))
names = by.pop("names", {})
tot = sum(len(v) for v in by.values())
for t in sorted(by):
    v = by[t]
    flag = "" if names[t][1] is None else f" | wording mismatch: {names[t][1]}"
    print(f"task {t}: {sum(v):2d}/{len(v):2d} = {100 * sum(v) / len(v):5.1f}% | {names[t][0]}{flag}")
print(f"OVERALL: {sum(sum(v) for v in by.values())}/{tot} = {100 * sum(sum(v) for v in by.values()) / max(tot, 1):.1f}%")
print(f"episodes with errors: {len(errors)}" + "".join(f"\n  {n}: {m}" for n, m in errors[:10]))
