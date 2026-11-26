"""Causal checks on the plate finding (after Stage 5). Nothing is trained; the model stays frozen.

  attention   tables from data/attention (collect.py --attention): attention from the 56 action positions to the
              patches at the plate's current spot vs the original spot, per layer, swap vs LIBERO-Spatial
  patch       12 swap queries carrying the bowl (tasks 0, 4, 5, 8; early / mid / late in the carry). For each, a twin
              observation is rendered from the same simulator state with the plate and the object it swapped with put
              back (plate at the original spot O). Then:
                baseline  model on the swap input vs on the twin input
                patching  image-token hidden states at one layer copied from one run into the other (both directions)
              Renders: results/causal/twins/*.png; numbers: results/causal/patching.csv
Usage: python causal.py attention | python causal.py patch [--renders-only]
"""
import argparse, copy, glob, os, sys

import numpy as np
import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "openvla-oft"))
import labels as L                                                                                # noqa: E402

CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
CC = CFG["causal"]
OUT = os.path.join(HERE, CC["out_dir"])
SPOT_XY = CFG["patch_check"]["spots"]


# ----------------------------------------------------------------------------------------------- attention
def attention():
    rows = []
    for suite in ("libero_spatial_swap", "libero_spatial"):
        for p in sorted(glob.glob(os.path.join(HERE, CC["attention_dir"], suite, "task*_ep*.pt"))):
            e = torch.load(p, weights_only=True)
            if e["n_queries"] == 0:
                continue
            g = e["attn_groups"]
            a = e["attn"].float().mean(2).numpy()                              # head mean: (Q, layers, groups)
            for q in range(e["n_queries"]):
                for layer in range(a.shape[1]):
                    rows.append(dict(suite=suite, task=e["task_id"], episode=e["episode"], q=q, layer=layer,
                                     phase="after" if bool(e["phase_after"][q]) else "before",
                                     verify=float(e["attn_verify_maxdiff"][q]),
                                     **{n: a[q, layer, i] for i, n in enumerate(g)}))
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    spot_of = {0: "A", 4: "A", 5: "A", 8: "A", 7: "C", 9: "C", 6: "B"}
    d["spot_now"] = np.where(d.suite == "libero_spatial", "O", d.task.map(spot_of))
    d["to_current_spot"] = [r[f"spot_{s}"] for s, r in zip(d.spot_now, d.to_dict("records"))]
    d["to_original_spot"] = d["spot_O"]
    d.to_csv(os.path.join(OUT, "attention_long.csv.gz"), index=False)
    print(f"{len(d)} rows; recomputed-vs-SDPA max |diff| {d.verify.max():.3g}")
    return d


# ----------------------------------------------------------------------------------------------- patching
def pick_queries(scale=False):
    """Carry = after grasp, gripper closed, bowl lifted >= 2 cm. Default (12 queries): per task, episode 0 early,
    1 mid, 2 late. scale (config causal.scale): every listed episode gives an early, a mid and a late query."""
    if scale:
        plan = [(int(t), ep, when) for t, eps in CC["scale"]["episodes"].items() for ep in eps
                for when in ("early", "mid", "late")]
    else:
        plan = [(t, ep, when) for t in CC["patch_tasks"]
                for ep, when in zip(CC["patch_episodes"], ("early", "mid", "late"))]
    picks = []
    for t, ep, when in plan:
        e = torch.load(os.path.join(HERE, CFG["collection"]["out_dir"], "libero_spatial_swap",
                                    f"task{t}_ep{ep:02d}.pt"), weights_only=True, mmap=True)
        carry = np.where(e["phase_after"].numpy() & e["gripper_close_next"].numpy()
                         & (e["bowl_lift"].numpy() >= CFG["labels"]["grasp_rule"]["lift_m"]))[0]
        q = int(carry[{"early": 0, "mid": len(carry) // 2, "late": len(carry) - 1}[when]])
        picks.append(dict(task=t, episode=ep, when=when, q=q, ref_chunk=e["chunk"][q].numpy()))
    return picks


def swapped_objects(task):
    """The two objects the LIBERO-PRO swap exchanged (the plate and one other), from the BDDL init regions."""
    from libero.libero import benchmark, get_libero_path
    tk = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(task)
    orig = os.path.join(get_libero_path("bddl_files"), tk.problem_folder, tk.bddl_file)
    swap = os.path.join(HERE, CFG["env"]["suites"]["libero_spatial_swap"]["bddl_dir"], tk.bddl_file)
    moved = L.wording_mismatch(tk.language, orig, swap)[2]
    assert len(moved) == 2 and "plate_1" in moved, moved
    return moved


def twin_obs(env, objs):
    """Same simulator state, but the two swapped objects exchange their xy positions (each keeps its own height and
    orientation), i.e. the plate goes back to its original spot. Rendered without stepping physics; the state is
    restored afterwards."""
    inner = env.env
    sim = inner.sim
    saved = sim.get_state().flatten().copy()
    j = {o: inner.objects_dict[o].joints[0] for o in objs}
    qp = {o: sim.data.get_joint_qpos(j[o]).copy() for o in objs}
    a, b = objs
    for o, src in ((a, b), (b, a)):
        new = qp[o].copy(); new[:2] = qp[src][:2]
        sim.data.set_joint_qpos(j[o], new)
    sim.forward()
    obs = inner._get_observations(force_update=True)
    sim.set_state_from_flattened(saved); sim.forward()
    return obs


class Patcher:
    """Forward hooks on chosen decoder layers: 'capture' keeps the image-token outputs, 'patch' overwrites chosen
    token positions of one layer's output with a captured copy."""

    def __init__(self, vla, layers):
        self.layers = {i: vla.language_model.model.layers[i] for i in layers}
        self.mode, self.store, self.src, self.layer, self.idx = None, {}, None, None, None
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in self.layers.items()]

    def _hook(self, i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.store[i] = h[0, 1:513].detach().clone()                # agentview 1..256, wrist 257..512
            elif self.mode == "patch" and i == self.layer:
                h[0, self.idx] = self.src[i][[k - 1 for k in self.idx]]
            return out
        return fn


def chunk_metrics(chunk, eef, plate_now, orig):
    """Net predicted move of the 8-step chunk (world xy; actions map to world axes, ~1.2 cm per summed unit,
    checked on Stage 3 data) and its direction relative to the plate's current spot and the original spot.
    grip_raw_mean: mean raw gripper output over the chunk (before process_action; ~1 = open, ~0 = close)."""
    d = chunk[:, :2].sum(0)
    end = eef[:2] + CC["metres_per_action_unit"] * d
    cos = lambda v: float(np.dot(d, v) / (np.linalg.norm(d) * np.linalg.norm(v) + 1e-9))
    return dict(cos_to_plate_now=cos(plate_now[:2] - eef[:2]), cos_to_original=cos(orig - eef[:2]),
                end_to_plate_now_cm=100 * float(np.linalg.norm(end - plate_now[:2])),
                end_to_original_cm=100 * float(np.linalg.norm(end - orig)), grip_raw_mean=float(chunk[:, 6].mean()),
                move_cm=100 * CC["metres_per_action_unit"]
                * float(np.linalg.norm(d)), first_dir=L.DIRECTIONS[L.direction_label(chunk[0, :3],
                                                                                   CFG["labels"]["direction_stay_threshold"])])


def patch(renders_only=False, scale=False):
    from collections import deque
    import collect as C
    from PIL import Image
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.robot_utils import get_action
    from libero.libero import benchmark
    out = os.path.join(OUT, CC["scale"]["out_subdir"]) if scale else OUT
    os.makedirs(os.path.join(out, "twins"), exist_ok=True)
    cfg, model, action_head, proprio_projector, processor, resize_size = C.load_model()
    run = lambda ob, desc: np.stack([np.asarray(a, np.float32) for a in get_action(
        cfg, model, copy.deepcopy(ob), desc, processor=processor, action_head=action_head,
        proprio_projector=proprio_projector, use_film=cfg.use_film)])
    pt = Patcher(model, CC["patch_layers"])
    orig_xy = np.array(CC["original_plate_world_xy"])
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    rows, picks = [], pick_queries(scale)
    for task in sorted({p["task"] for p in picks}):
        tk = suite.get_task(task)
        env, desc, _ = C.make_env("libero_spatial_swap", tk, task)
        states = C.initial_states(suite, "libero_spatial_swap", tk, task)
        objs = swapped_objects(task)
        for pk in [p for p in picks if p["task"] == task]:
            C.set_seed_everywhere(cfg.seed + 1000 * task + pk["episode"])
            env.reset()
            obs = env.set_init_state(states[pk["episode"]])
            for _ in range(cfg.num_steps_wait):
                obs, *_ = env.step(get_libero_dummy_action(cfg.model_family))
            queue, q = deque(maxlen=cfg.num_open_loop_steps), -1
            while True:                                                       # official loop up to the chosen query
                if len(queue) == 0:
                    q += 1
                    ob, img = C.prepare_observation(obs, resize_size)
                    if q == pk["q"]:
                        break
                    queue.extend(run(ob, desc))
                obs, *_ = env.step(C.process_action(queue.popleft(), cfg.model_family).tolist())
            tw = twin_obs(env, objs)
            ob_t, img_t = C.prepare_observation(tw, resize_size)
            name = f"task{task}_ep{pk['episode']:02d}_q{q:02d}_{pk['when']}"
            wrist = lambda o: np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])
            Image.fromarray(np.concatenate([np.concatenate([img, img_t], 1),
                                            np.concatenate([wrist(obs), wrist(tw)], 1)], 0)).save(
                os.path.join(out, "twins", f"{name}.png"))
            eef, plate_now = obs["robot0_eef_pos"].astype(np.float32), obs["plate_1_pos"].astype(np.float32)
            base = dict(task=task, episode=pk["episode"], when=pk["when"], q=q, eef_x=eef[0], eef_y=eef[1])
            pt.mode = "capture"; ch_s = run(ob, desc); st_s = dict(pt.store)
            ch_t = run(ob_t, desc); st_t = dict(pt.store); pt.mode = None
            repro = float(np.abs(ch_s - pk["ref_chunk"]).max())               # same query as in Stage 3?
            for cond, ch in (("swap input", ch_s), ("twin input (plate at O)", ch_t)):
                rows.append(dict(base, condition=cond, layer=None, tokens=None, repro_vs_stage3=repro,
                                 chunk_change=float(np.abs(ch - ch_s).max()), **chunk_metrics(ch, eef, plate_now, orig_xy)))
            if renders_only:
                continue
            sets = {"agentview": list(range(1, 257)), "all image (agentview + wrist)": list(range(1, 513)),
                    "spots A + O (3x3 each)": L.patch_block(L.patch_token(SPOT_XY["A"]))
                    + L.patch_block(L.patch_token(SPOT_XY["O"]))}
            for layer in CC["patch_layers"]:
                for tok_name, idx in sets.items():
                    for direction, ob_in, src, ref in (("twin -> swap run", ob, st_t, ch_s),
                                                        ("swap -> twin run", ob_t, st_s, ch_t)):
                        pt.mode, pt.layer, pt.idx, pt.src = "patch", layer, idx, src
                        ch = run(ob_in, desc); pt.mode = None
                        rows.append(dict(base, condition=direction, layer=layer, tokens=tok_name,
                                         chunk_change=float(np.abs(ch - ref).max()),
                                         **chunk_metrics(ch, eef, plate_now, orig_xy)))
            print(f"{name}: done ({len(rows)} rows)", flush=True)
        env.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "patching.csv" if not renders_only else "twin_baseline.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["attention", "patch"])
    ap.add_argument("--renders-only", action="store_true")
    ap.add_argument("--scale", action="store_true", help="the larger query set in config causal.scale")
    a = ap.parse_args()
    attention() if a.what == "attention" else patch(a.renders_only, a.scale)
