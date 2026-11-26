"""Collect OpenVLA-OFT hidden states + labels from LIBERO rollouts. The model is frozen (eval mode, inference_mode).

The episode loop is the official one (openvla-oft experiments/robot/libero/run_libero_eval.py: run_episode), using its
own functions for model loading, observation preprocessing, action querying and action post-processing. Added around
it: forward hooks that record hidden states at each model query, and simulator labels (labels.py).

One file per episode: <out>/<suite>/task<k>_ep<e>.pt (torch.save of tensors / lists only).
Usage:
  python collect.py --suite libero_spatial --tasks 0 --episodes 0 1 --check      # Stage 2 dry run
  python collect.py --suite libero_spatial_swap --tasks 0 1 2 --n-episodes 50     # full collection
"""
import argparse, copy, glob, json, os, platform, shutil, subprocess, sys, time
from collections import deque

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OFT = os.path.join(HERE, "third_party", "openvla-oft")
sys.path.insert(0, OFT)

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env    # noqa: E402
from experiments.robot.libero.run_libero_eval import (GenerateConfig, TASK_MAX_STEPS, initialize_model,  # noqa: E402
                                                      prepare_observation, process_action)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere       # noqa: E402
from libero.libero import benchmark, get_libero_path                                              # noqa: E402
from libero.libero.envs import OffScreenRenderEnv                                                 # noqa: E402

import labels as L                                                                                # noqa: E402

CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
POS_NAMES = ["act56", "act7", "last"]


class HiddenRecorder:
    """Forward hooks on chosen decoder layers (+ final RMSNorm). For each model query keeps, per hook point,
    [mean over the 56 action-head positions, mean over the first 7, last token] as bf16 on CPU.
    Sequence = <BOS> | 513 patch/proprio | P prompt | 56 action slots | </s>; the action head reads [S-58, S-2)."""

    def __init__(self, vla, layers, final_norm, keep_full_norm=False):
        lm = vla.language_model.model
        self.points = [f"L{i}" for i in layers] + (["norm"] if final_norm else [])
        self.mods = [lm.layers[i] for i in layers] + ([lm.norm] if final_norm else [])
        self.keep_full_norm, self.handles, self.buf, self.full_norm = keep_full_norm, [], {}, None
        self.patch_idx, self.pbuf = None, {}          # patch check: sequence indices of agentview patch tokens

    def attach(self):
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in zip(self.points, self.mods)]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, name):
        def fn(module, inputs, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]          # (S, D), batch size is 1
            S = h.shape[0]; a0 = S - 58
            assert a0 > 513, f"unexpected sequence length {S}"
            hf = h.float()
            self.buf[name] = torch.stack([hf[a0:a0 + 56].mean(0), hf[a0:a0 + 7].mean(0), hf[S - 1]]
                                         ).to(torch.bfloat16).cpu()
            if self.patch_idx is not None:            # chosen patch tokens + mean over all 256 agentview patches
                self.pbuf[name] = torch.cat([hf[self.patch_idx], hf[1:257].mean(0, keepdim=True)]
                                            ).to(torch.bfloat16).cpu()
            if name == "norm" and self.keep_full_norm:
                self.full_norm = h[a0:a0 + 56].detach().clone()
        return fn

    def pop(self):
        assert set(self.buf) == set(self.points), f"missing hook outputs: {set(self.points) - set(self.buf)}"
        out = torch.stack([self.buf[p] for p in self.points], dim=1)              # (3 positions, n_points, D)
        self.buf = {}
        return out

    def pop_patch(self):
        out = torch.stack([self.pbuf[p] for p in self.points], dim=1)             # (n_tokens + 1, n_points, D)
        self.pbuf = {}
        return out


class AttentionRecorder:
    """Attention from the 56 action positions to groups of tokens, per decoder layer and head. Computed inside the
    model's own attention call from the same q and k: the OpenVLA-OFT fork calls SDPA with is_causal=False and, for one
    unpadded sequence, an all-zero mask, so the weights are softmax(q k^T / sqrt(d)) over every token. The eager
    attention path is NOT bidirectional in the fork, so it cannot be used for this. Per layer: mean over the 56 query
    rows of the total weight on each group, (n_heads, n_groups)."""
    FIXED = {"bos": [0], "agentview": list(range(1, 257)), "wrist": list(range(257, 513)), "proprio": [513]}

    def __init__(self, vla, detail=False):
        self.layers = vla.language_model.model.layers
        self.regions, self.layer, self.out, self.verify, self.handles = None, None, {}, {}, []
        # detail: also per-token attention (head mean, over the whole sequence) and value-norm-weighted group sums,
        # i.e. sum_j a_ij ||v_j|| per group, renormalized over all tokens (o_proj ignored)
        self.detail, self.tok, self.vw = detail, {}, {}
        self.names = list(self.FIXED) + ["text", "action", "tail"]

    def attach(self):
        self._sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped
        for i, m in enumerate(self.layers):
            self.handles.append(m.register_forward_pre_hook(lambda mod, inp, i=i: setattr(self, "layer", i)))
            self.handles.append(m.register_forward_hook(lambda mod, inp, out: setattr(self, "layer", None)))

    def detach(self):
        torch.nn.functional.scaled_dot_product_attention = self._sdpa
        for h in self.handles:
            h.remove()
        self.handles = []

    def _wrapped(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        out = self._sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
        if self.layer is None or self.regions is None:                   # vision backbones, or recording off
            return out
        assert not is_causal and q.shape[0] == 1
        S = q.shape[2]; a0 = S - 58
        logits = q[0, :, a0:a0 + 56].float() @ k[0].float().transpose(-1, -2) / q.shape[-1] ** 0.5   # (H, 56, S)
        if attn_mask is not None:
            logits = logits + attn_mask[0, :, a0:a0 + 56].float()
        p = logits.softmax(-1)
        groups = dict(self.FIXED, text=list(range(514, a0)), action=list(range(a0, a0 + 56)),
                      tail=list(range(a0 + 56, S)), **self.regions)
        self.out[self.layer] = torch.stack([p[..., g].sum(-1).mean(-1) for g in groups.values()], -1).half().cpu()
        if self.detail:
            self.tok[self.layer] = p.mean((0, 1)).half().cpu()                          # (S,)
            pv = p * v[0].float().norm(dim=-1)[:, None, :]                              # (H, 56, S)
            pv = pv / pv.sum(-1, keepdim=True)
            self.vw[self.layer] = torch.stack([pv[..., g].sum(-1).mean(-1) for g in groups.values()], -1).half().cpu()
        if self.layer in (0, len(self.layers) - 1):                     # recomputed weights reproduce SDPA's output
            self.verify[self.layer] = float((p @ v[0].float() - out[0, :, a0:a0 + 56].float()).abs().max())
        return out

    def pop(self):
        assert len(self.out) == len(self.layers), f"attention from {len(self.out)} layers"
        a = torch.stack([self.out[i] for i in range(len(self.layers))])          # (layers, heads, groups)
        ver = max(self.verify.values())
        extra = {}
        if self.detail:
            extra = dict(attn_tok=torch.stack([self.tok[i] for i in range(len(self.layers))]),
                         attn_vnorm=torch.stack([self.vw[i] for i in range(len(self.layers))]))
        self.out, self.verify, self.tok, self.vw = {}, {}, {}, {}
        return a, ver, extra


def make_env(suite_name, task, task_id):
    """Original suite: the official get_libero_env. Swap suite: same, but the LIBERO-PRO BDDL file."""
    if suite_name == "libero_spatial":
        env, desc = get_libero_env(task, "openvla", resolution=CFG["env"]["env_img_res"])
        return env, desc, None
    bddl = os.path.join(HERE, CFG["env"]["suites"][suite_name]["bddl_dir"], task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CFG["env"]["env_img_res"],
                             camera_widths=CFG["env"]["env_img_res"])
    env.seed(CFG["env"]["env_seed"])                 # as in get_libero_env
    return env, task.language, bddl


def initial_states(suite, suite_name, task, task_id):
    if suite_name == "libero_spatial":
        return suite.get_task_init_states(task_id)   # LIBERO's own files (official eval path)
    f = os.path.join(HERE, CFG["env"]["suites"][suite_name]["init_states"],
                     task.bddl_file.replace(".bddl", ".pruned_init"))
    return L.load_init_states_safe(f)


def run_episode(cfg, env, desc, model, processor, action_head, proprio_projector, resize_size, init_state, rec,
                check=False, record_frames=False, patch_stride=None, rng=None, att=None):
    """Official run_episode + label/hidden recording. Returns (rows, trace, success, error, extra).
    patch_stride: patch check mode -- record patch tokens and keep only every patch_stride-th after-grasp query."""
    spots = [L.patch_token(xy) for xy in CFG["patch_check"]["spots"].values()] if patch_stride else None
    n_after = 0
    max_steps, wait = TASK_MAX_STEPS[cfg.task_suite_name], cfg.num_steps_wait
    target, place = env.env.obj_of_interest[0], env.env.obj_of_interest[1]
    env.reset()
    obs = env.set_init_state(init_state)
    sim = env.env.sim                                    # read after reset: reset can rebuild the simulator
    queue = deque(maxlen=cfg.num_open_loop_steps)
    rows, trace, extra = [], dict(t=[], bowl_lift=[], contact=[], gripper_close=[], grasped=[]), dict(checks=[])
    video = extra.setdefault("video", []) if record_frames else None
    tracker, last_close, success, error, t = None, False, False, None, 0
    try:
        while t < max_steps + wait:
            if t < wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            if tracker is None:                                          # resting height after settling
                tracker = L.GraspTracker(obs[f"{target}_pos"][2], CFG["labels"]["grasp_rule"]["lift_m"])
            observation, img = prepare_observation(obs, resize_size)
            if video is not None:                                        # same frames the official replay video uses
                video.append(np.ascontiguousarray(img[::2, ::2]))       # 256 -> 128 px
            if len(queue) == 0:
                obs_for_check = copy.deepcopy(observation) if check else None   # get_vla_action normalizes
                if patch_stride:                                                 # tokens chosen before the forward
                    plate_tok = L.patch_token(L.project(sim, obs[f"{place}_pos"][None])[0])
                    rand_tok = 1 + int(rng.choice([p for p in range(256) if p != plate_tok - 1]))
                    rec.patch_idx = spots + [plate_tok, rand_tok]
                if att is not None:                                              # 3x3 patch blocks per query
                    pxy = L.project(sim, np.stack([obs[f"{place}_pos"], obs[f"{target}_pos"]]))
                    att.regions = {**{f"spot_{n}": L.patch_block(L.patch_token(xy))
                                      for n, xy in CFG["patch_check"]["spots"].items()},
                                   "plate_now": L.patch_block(L.patch_token(pxy[0])),
                                   "bowl_now": L.patch_block(L.patch_token(pxy[1]))}
                tq = time.time()                                                 # observation["state"] in place
                actions = get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                                     proprio_projector=proprio_projector, use_film=cfg.use_film)
                dt = time.time() - tq
                hidden = rec.pop()
                patch = rec.pop_patch() if patch_stride else None
                chunk = np.stack([np.asarray(a, np.float32) for a in actions])  # (8, 7) unnormalized, as returned
                if check:
                    extra["checks"].append(check_query(cfg, model, obs_for_check, desc, processor, action_head,
                                                       proprio_projector, rec, chunk))
                bowl, plate = obs[f"{target}_pos"].astype(np.float32), obs[f"{place}_pos"].astype(np.float32)
                xy = L.project(sim, np.stack([bowl, plate]))
                contact = L.gripper_contact(env, target)
                nxt = process_action(chunk[0].copy(), cfg.model_family)
                rows.append(dict(
                    t=t - wait, hidden=hidden, chunk=torch.from_numpy(chunk), query_s=dt,
                    next_xyz=torch.from_numpy(chunk[0, :3]), gripper_close_next=bool(nxt[6] > 0),
                    contact=contact, contact_plate=L.gripper_contact(env, place),
                    grasped_now=tracker.is_grasped(contact, last_close, bowl[2]), phase_after=tracker.phase_after,
                    bowl_xyz=torch.from_numpy(bowl), plate_xyz=torch.from_numpy(plate),
                    bowl_xy=torch.from_numpy(xy[0]), plate_xy=torch.from_numpy(xy[1]),
                    bowl_xy224=torch.from_numpy(L.to_policy_frame(xy[0])),
                    plate_xy224=torch.from_numpy(L.to_policy_frame(xy[1])),
                    target_xy=torch.from_numpy(xy[1] if tracker.phase_after else xy[0]),
                    eef_pos=torch.from_numpy(obs["robot0_eef_pos"].astype(np.float32)),
                    gripper_qpos=torch.from_numpy(obs["robot0_gripper_qpos"].astype(np.float32)),
                    bowl_lift=float(bowl[2] - tracker.rest_z)))
                if att is not None:
                    a, ver, extra_att = att.pop()
                    rows[-1].update(attn=a, attn_verify_maxdiff=ver, **extra_att)
                if patch_stride:
                    keep = tracker.phase_after and n_after % patch_stride == 0
                    n_after += tracker.phase_after
                    rows[-1].update(patch_hidden=patch, patch_tokens=torch.tensor(rec.patch_idx), keep=keep)
                if check and len(rows) in (1, 8, 14, 20):
                    extra.setdefault("frames", []).append((len(rows) - 1, img.copy(), xy.copy()))
                queue.extend(actions)
            action = process_action(queue.popleft(), cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            last_close = bool(action[6] > 0)
            contact = L.gripper_contact(env, target)
            z = obs[f"{target}_pos"][2]
            g = tracker.update(contact, last_close, z)
            for k, v in zip(trace, (t - wait, z - tracker.rest_z, contact, last_close, g)):
                trace[k].append(v)
            if done:
                success = True
                break
            t += 1
    except Exception as e:                                               # official code logs and continues
        error = f"{type(e).__name__}: {e}"
    return rows, trace, success, error, extra


def check_query(cfg, model, observation, desc, processor, action_head, proprio_projector, rec, chunk):
    """(1) Remove the hooks, query again on the identical observation, compare actions exactly.
    (2) Feed the recorded final-norm hidden states at the 56 positions to the action head and check that it
    reproduces the model's actions, i.e. that the recorded positions are the ones the head reads."""
    full = rec.full_norm
    rec.detach()
    again = np.stack([np.asarray(a, np.float32) for a in get_action(
        cfg, model, observation, desc, processor=processor, action_head=action_head,
        proprio_projector=proprio_projector, use_film=cfg.use_film)])
    rec.attach()
    with torch.inference_mode():
        norm_actions = action_head.predict_action(full[None]).reshape(8, 7).float().cpu().numpy()
    from_head = model._unnormalize_actions(norm_actions, cfg.unnorm_key)
    return dict(hooks_vs_nohooks_maxdiff=float(np.abs(again - chunk).max()),
                head_on_recorded_maxdiff=float(np.abs(np.asarray(from_head, np.float32) - chunk).max()))


def save_video(frames, path, fps=30):
    import imageio
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=5, macro_block_size=16)
    for f in frames:
        w.append_data(f)
    w.close()


def to_episode(rows, trace, meta):
    keys = [k for k in rows[0] if k != "hidden"] if rows else []
    ep = dict(meta, n_queries=len(rows),
              hidden=torch.stack([r["hidden"] for r in rows]) if rows else torch.empty(0),
              hidden_points=meta["hidden_points"], hidden_positions=POS_NAMES,
              trace={k: torch.tensor(np.asarray(v, dtype=np.float32)) for k, v in trace.items()})
    for k in keys:
        v = [r[k] for r in rows]
        ep[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
    return ep


def load_model():
    """Official GenerateConfig + initialize_model with the settings in config.yaml; frozen (eval, no grad)."""
    m = CFG["model"]
    cfg = GenerateConfig(pretrained_checkpoint=m["checkpoint"], use_l1_regression=m["use_l1_regression"],
                         use_diffusion=m["use_diffusion"], use_film=m["use_film"],
                         num_images_in_input=m["num_images_in_input"], use_proprio=m["use_proprio"],
                         center_crop=m["center_crop"], num_open_loop_steps=m["num_open_loop_steps"],
                         lora_rank=m["lora_rank"], task_suite_name="libero_spatial",   # unnorm key + max steps
                         num_steps_wait=CFG["env"]["num_steps_wait"], env_img_res=CFG["env"]["env_img_res"],
                         seed=CFG["env"]["seed"])
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, _, processor = initialize_model(cfg)
    model.eval(); action_head.eval(); proprio_projector.eval()
    for mod in (model, action_head, proprio_projector):
        for p in mod.parameters():
            p.requires_grad_(False)
    return cfg, model, action_head, proprio_projector, processor, get_image_resize_size(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial", choices=list(CFG["env"]["suites"]))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--episodes", type=int, nargs="+", default=None, help="initial-state indices")
    ap.add_argument("--n-episodes", type=int, default=CFG["collection"]["episodes_per_task"])
    ap.add_argument("--out", default=os.path.join(HERE, CFG["collection"]["out_dir"]))
    ap.add_argument("--check", action="store_true", help="hook-equivalence + action-head-position checks per query")
    ap.add_argument("--no-resume", action="store_true", help="re-run episodes whose file already exists")
    ap.add_argument("--min-free-gb", type=float, default=0.5, help="stop cleanly below this much free disk")
    ap.add_argument("--video-dir", default=None, help="save 1 success + 1 failure video per task here")
    ap.add_argument("--attention", action="store_true",
                    help="record attention from the 56 action positions to token groups (hidden states not saved)")
    ap.add_argument("--attention-detail", action="store_true",
                    help="with --attention: also per-token attention and value-norm-weighted group sums")
    ap.add_argument("--patch-stride", type=int, default=None,
                    help="patch check: save patch tokens (config patch_check) at every n-th after-grasp query, and "
                         "keep only those queries in the file")
    args = ap.parse_args()

    cfg, model, action_head, proprio_projector, processor, resize_size = load_model()
    rec = HiddenRecorder(model, CFG["hooks"]["layers"], CFG["hooks"]["final_norm"], keep_full_norm=args.check)
    rec.attach()
    att = AttentionRecorder(model, detail=args.attention_detail) if args.attention else None
    if att:
        att.attach()

    out_dir = os.path.join(args.out, args.suite); os.makedirs(out_dir, exist_ok=True)
    run_info = dict(argv=sys.argv, config=CFG, unnorm_key=cfg.unnorm_key, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                    openvla_oft_commit=subprocess.run(["git", "-C", OFT, "rev-parse", "HEAD"], capture_output=True,
                                                      text=True).stdout.strip(), started=time.ctime())
    json.dump(run_info, open(os.path.join(out_dir, f"run_{int(time.time())}.json"), "w"), indent=1, default=str)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        env, desc, swap_bddl = make_env(args.suite, task, task_id)
        states = initial_states(suite, args.suite, task, task_id)
        mismatch = None
        if swap_bddl:
            orig = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            mismatch = L.wording_mismatch(task.language, orig, swap_bddl)[0]
        eps = args.episodes if args.episodes is not None else range(min(args.n_episodes, len(states)))
        vdir = os.path.join(args.video_dir, args.suite) if args.video_dir else None
        if vdir:
            os.makedirs(vdir, exist_ok=True)
        for e in eps:
            path = os.path.join(out_dir, f"task{task_id}_ep{e:02d}.pt")
            if not args.no_resume and os.path.exists(path):
                continue
            free_gb = shutil.disk_usage(out_dir).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"STOPPED: only {free_gb:.2f} GB free (< {args.min_free_gb} GB) before {args.suite} task {task_id} "
                      f"episode {e}. Everything saved so far is complete; re-run the same command to resume.", flush=True)
                env.close()
                sys.exit(3)
            have = {k: bool(glob.glob(os.path.join(vdir, f"task{task_id}_{k}_ep*.mp4"))) for k in ("success", "failure")} \
                if vdir else {"success": True, "failure": True}
            set_seed_everywhere(cfg.seed + 1000 * task_id + e)          # per-episode seed, logged
            t0 = time.time()
            rows, trace, success, error, extra = run_episode(cfg, env, desc, model, processor, action_head,
                                                             proprio_projector, resize_size, states[e], rec,
                                                             check=args.check,
                                                             record_frames=not (have["success"] and have["failure"]),
                                                             patch_stride=args.patch_stride,
                                                             rng=np.random.default_rng(cfg.seed + 1000 * task_id + e),
                                                             att=att)
            if att:                                                        # attention mode: hidden states not kept
                for r in rows:
                    r["hidden"] = torch.empty(0)
            if args.patch_stride:                                           # keep only the selected queries
                for i, r in enumerate(rows):
                    r["q_index"] = i
                rows = [{k: v for k, v in r.items() if k != "keep"} for r in rows if r["keep"]]
            meta = dict(suite=args.suite, task_id=task_id, episode=e, episode_id=f"{args.suite}/task{task_id}_ep{e:02d}",
                        task=desc, target=env.env.obj_of_interest[0], place=env.env.obj_of_interest[1],
                        success=success, error=error, wording_mismatch=mismatch, seconds=time.time() - t0,
                        seed=cfg.seed + 1000 * task_id + e, hidden_points=rec.points,
                        patch_order=(list(CFG["patch_check"]["spots"]) + ["plate_now", "random", "mean_all"])
                        if args.patch_stride else None,
                        attn_groups=(att.names + list(att.regions)) if att and att.regions else None,
                        checks=extra["checks"])
            tmp = path + ".tmp"
            torch.save(to_episode(rows, trace, meta), tmp)
            os.replace(tmp, path)                       # atomic: a stopped run never leaves a half-written file
            kind = "success" if success else "failure"
            if extra.get("video") and not error and not have[kind]:
                save_video(extra["video"], os.path.join(vdir, f"task{task_id}_{kind}_ep{e:02d}.mp4"))
            if extra.get("frames"):
                np.savez_compressed(path.replace(".pt", "_frames.npz"),
                                    q=np.array([f[0] for f in extra["frames"]]),
                                    img=np.stack([f[1] for f in extra["frames"]]),
                                    xy=np.stack([f[2] for f in extra["frames"]]))
            print(f"{meta['episode_id']}: success={success} queries={len(rows)} steps={len(trace['t'])} "
                  f"{meta['seconds']:.1f}s size={os.path.getsize(path) / 1e6:.2f} MB"
                  + (f" ERROR {error}" if error else ""), flush=True)
            if extra["checks"]:
                c = extra["checks"]
                print(f"  checks over {len(c)} queries: hooks-vs-no-hooks max |diff| "
                      f"{max(x['hooks_vs_nohooks_maxdiff'] for x in c):.3g}, action head on recorded positions "
                      f"max |diff| {max(x['head_on_recorded_maxdiff'] for x in c):.3g}", flush=True)
        env.close()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
