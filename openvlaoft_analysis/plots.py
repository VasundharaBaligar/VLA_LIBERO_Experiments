"""Figures for the probing study.
  grasp-traces <episode .pt files>   bowl lift, contact, gripper command and the grasp rule over each episode
  projection   <episode .pt file>    bowl / plate projections drawn on saved policy frames (needs *_frames.npz)
  stage5                             score-vs-layer figures, plate-test figures and the summary table from the
                                     Stage 4 tables in results/stage4 (written to results/stage5)
  detail                             confusion matrices, Q5 scatter, MLP training curves, score vs layer with seed
                                     error bars, presence heatmaps, and one PDF of them (needs the best-point
                                     prediction files and `probes.py --curves`)
Usage: python plots.py grasp-traces data/dryrun/libero_spatial/task0_ep0*.pt --out results/stage2_grasp_traces.png
       python plots.py stage5 --out results/stage5
       python plots.py detail --out results/stage5/plots_detail --preds <folder with the prediction files>"""
import argparse, os

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # categorical slots, fixed order


def load(path):
    return torch.load(path, weights_only=True)


def grasp_traces(paths, out, lift_m=0.02):
    eps = [load(p) for p in paths]
    fig, axes = plt.subplots(len(eps), 1, figsize=(9, 2.6 * len(eps)), squeeze=False, sharex=True)
    for ax, ep in zip(axes[:, 0], eps):
        tr = ep["trace"]
        t, lift = tr["t"].numpy(), tr["bowl_lift"].numpy() * 100
        ax.plot(t, lift, color=C[0], lw=2, label="bowl height above rest (cm)")
        ax.axhline(lift_m * 100, color=MUTED, lw=1, ls="--")
        ax.text(t[-1], lift_m * 100, f" {lift_m * 100:.0f} cm", va="center", fontsize=8, color=MUTED)
        top = max(lift.max(), lift_m * 100) * 1.15 + 0.5
        for k, (key, col, lab) in enumerate([("contact", C[1], "gripper-bowl contact"),
                                              ("gripper_close", C[3], "gripper command = close"),
                                              ("grasped", C[2], "grasp rule true")]):
            on = tr[key].numpy() > 0.5
            y = -1.2 * (k + 1)
            ax.fill_between(t, y - 0.45, y + 0.45, where=on, color=col, step="mid", lw=0, label=lab)
        first = np.argmax(tr["grasped"].numpy() > 0.5) if (tr["grasped"] > 0.5).any() else None
        if first is not None:
            ax.axvline(t[first], color=C[2], lw=1)
            ax.text(t[first], top, f" phase -> after grasp (step {int(t[first])})", fontsize=8, color=INK, va="top")
        ax.set_ylim(-4.3, top); ax.set_yticks([0, round(top / 2), int(top)])
        ax.set_title(f"{ep['episode_id']}  |  success={ep['success']}  |  {ep['task']}", loc="left", fontsize=9,
                     color=INK)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32))
    axes[-1, 0].set_xlabel("simulator step (after the 10 settle steps)", color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def projection(path, out):
    ep, fr = load(path), np.load(path.replace(".pt", "_frames.npz"))
    n = len(fr["q"])
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.3), squeeze=False)
    for ax, q, img, xy in zip(axes[0], fr["q"], fr["img"], fr["xy"]):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        after = bool(ep["phase_after"][q])
        for (x, y), col, lab in zip(xy, (C[0], C[1]), ("bowl", "plate")):
            ax.scatter([x], [y], s=70, facecolor="none", edgecolor=col, lw=2.2, label=lab)
        tx, ty = ep["target_xy"][q].tolist()
        ax.scatter([tx], [ty], s=18, color="white", edgecolor=INK, lw=0.8, label="target label")
        ax.set_title(f"query {q} (step {int(ep['t'][q])})\nphase: {'after' if after else 'before'} grasp",
                     fontsize=9, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------- Stage 5
PTS = ["L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L31", "norm"]
POS = {"act56": "mean of 56 action positions", "act7": "mean of first 7 action positions", "last": "last token (</s>)"}
SUITE_NAMES = {"libero_spatial": "held-out LIBERO-Spatial task", "libero_spatial_swap": "same task, LIBERO-PRO swap"}
VARIANTS = [("lin", C[0], "-", "linear"), ("mlp", C[1], "-", "MLP"),
            ("lin_shuf", C[0], "--", "linear, shuffled labels"), ("mlp_shuf", C[1], "--", "MLP, shuffled labels")]
TARGETS = {"direction": ("Q1 direction of the next action (stay < 0.10)", 1 / 7),
           "direction_030": ("Q1 direction, sensitivity check (stay < 0.30)", 1 / 7),
           "distance": ("Q2 distance of the next action (8 quantile bins)", 1 / 8),
           "gripper": ("Q3 next gripper command = close", 1 / 2),
           "contact": ("Q4 gripper-bowl contact", 1 / 2),
           "target_xy": ("Q5 target position in the image (bowl before grasp, plate after)", None)}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _lines(ax, d, chance=None, control=None):
    x = np.arange(len(PTS))
    for v, col, ls, lab in VARIANTS:
        z = d[d.variant == v].set_index("point").reindex(PTS)
        ax.plot(x, z["mean"], color=col, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4, label=lab)
        if ls == "-":
            ax.fill_between(x, z["mean"] - z["std_splits"], z["mean"] + z["std_splits"], color=col, alpha=0.12, lw=0)
    if chance is not None:
        ax.axhline(chance, color=MUTED, lw=1, ls=":", label="chance")
    if control is not None:
        z = d[d.variant == "task_id"].set_index("point").reindex(PTS)
        ax.plot(x, z["mean"], color=MUTED, lw=1.5, ls=":", label="task-ID control")
    ax.set_xticks(x); ax.set_xticklabels(PTS, rotation=0)
    _style(ax)


def score_vs_layer(d, target, out, phase="all", positions=tuple(POS), cols=None):
    """Rows: suites; columns: token positions (or phases when cols='phase'). 4 lines + chance / task-ID control."""
    title, chance = TARGETS[target]
    d = d[d.target == target]
    reg = d.metric.iloc[0] == "px_err"
    panels = [(p, phase) for p in positions] if cols != "phase" else [(positions[0], ph) for ph in ("before", "after")]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 6.6), sharex=True, sharey=True, squeeze=False)
    for i, suite in enumerate(SUITE_NAMES):
        for j, (pos, ph) in enumerate(panels):
            ax = axes[i, j]
            _lines(ax, d[(d.suite == suite) & (d.pos == pos) & (d.phase == ph)], None if reg else chance,
                   control=True if reg else None)
            head = POS[pos] if cols != "phase" else f"{POS[pos]}, {ph} grasp"
            ax.set_title(f"{head}\n{SUITE_NAMES[suite]}", fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("mean pixel error (lower is better)" if reg else "balanced accuracy", color=MUTED)
    if not reg:
        axes[0, 0].set_ylim(0, 1.02)
    axes[-1, 0].set_xlabel("hook point (decoder layer output; norm = final RMSNorm)", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0.01, 0.95))
    fig.suptitle(title + ("" if phase == "all" and cols != "phase" else "  |  split by phase"), x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plate_presence(scores, out):
    """Presence probe across layers: score at the plate's true spot (A tasks, C tasks, B task) and at O."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    series = [("A tasks 0, 4, 5, 8: score at A (plate there)", C[0], "-", "libero_spatial_swap", [0, 4, 5, 8], "A"),
              ("C tasks 7, 9: score at C (plate there)", C[1], "-", "libero_spatial_swap", [7, 9], "C"),
              ("swap tasks: score at O (plate gone)", C[2], "--", "libero_spatial_swap", [0, 4, 5, 6, 7, 8, 9], "O"),
              ("Spatial tasks: score at O (plate there)", C[2], "-", "libero_spatial", [0, 4, 5, 6, 7, 8, 9], "O"),
              ("task 6: score at B (untestable, no B examples in training)", C[3], ":", "libero_spatial_swap", [6], "B")]
    x = np.arange(len(PTS))
    for ax, v, name in zip(axes, ("lin", "mlp"), ("linear probe", "MLP probe")):
        for lab, col, ls, suite, tasks, spot in series:
            z = scores[(scores.variant == v) & (scores.suite == suite) & scores.task.isin(tasks)]
            y = z.groupby("point")[spot].mean().reindex(PTS)
            ax.plot(x, y, color=col, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4, label=lab)
        sh = scores[(scores.variant == v + "_shuf") & (scores.suite == "libero_spatial_swap")]
        ax.plot(x, sh.groupby("point")[["A", "B", "C", "O"]].mean().mean(1).reindex(PTS), color=MUTED, lw=1.2,
                ls=":", label="shuffled-label control (mean over spots)")
        ax.set_xticks(x); ax.set_xticklabels(PTS); ax.set_ylim(0, 1.02); _style(ax)
        ax.set_title(f"plate-present score of the spot patch token, {name}", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("mean P(plate present)", color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.suptitle(" ", fontsize=4)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plate_probe(plate, out, exp="plate_swap", pos="act7"):
    """Action-position plate probe: error to the plate's current position vs the shuffled control, per spot group."""
    grp = {"A (tasks 0, 4, 5, 8)": [0, 4, 5, 8], "C (tasks 7, 9)": [7, 9], "B (task 6)": [6],
           "tasks 1, 2, 3 (separate)": [1, 2, 3]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True)
    x = np.arange(len(PTS))
    for ax, (g, tasks) in zip(axes, grp.items()):
        z = plate[(plate.exp == exp) & (plate.target == "plate_xy") & (plate.pos == pos) & plate.task.isin(tasks)]
        for v, col, ls, lab in VARIANTS:
            y = z[z.variant == v].groupby("point")["to_plate_now"].mean().reindex(PTS)
            ax.plot(x, y, color=col, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4, label=lab)
        ax.set_title(f"plate now at {g}", fontsize=9, color=INK, loc="left")
        ax.set_xticks(x); ax.set_xticklabels(PTS, fontsize=7); _style(ax)
    axes[0].set_ylabel("error to the plate's current position (px)", color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def summary_table(d, out):
    """Best hook point / position per question (highest held-out Spatial score of linear or MLP), with linear vs MLP,
    shuffled control, selectivity and the normal -> swap change. Q5: pixel error (lower is better)."""
    rows = []
    for target, (title, chance) in TARGETS.items():
        z = d[(d.target == target) & (d.phase == "all")]
        if z.empty:
            continue
        reg = z.metric.iloc[0] == "px_err"
        sp = z[(z.suite == "libero_spatial") & z.variant.isin(["lin", "mlp"])]
        b = sp.sort_values("mean", ascending=reg).iloc[0]
        k = z.set_index(["point", "pos", "variant", "suite"])["mean"]
        g = lambda v, s="libero_spatial": k.get((b.point, b.pos, v, s), np.nan)
        best_v = "mlp" if (g("mlp") < g("lin") if reg else g("mlp") > g("lin")) else "lin"
        shuf = g(best_v + "_shuf")
        rows.append({"question": title, "metric": "px error" if reg else "balanced acc", "chance": chance,
                     "best point": f"{b.point} / {b.pos}", "linear": g("lin"), "MLP": g("mlp"),
                     "best probe": "MLP" if best_v == "mlp" else "linear", "shuffled (same probe)": shuf,
                     "selectivity": (shuf - g(best_v)) if reg else (g(best_v) - shuf),
                     "task-ID control": g("task_id") if reg else np.nan,
                     "swap": g(best_v, "libero_spatial_swap"),
                     "normal -> swap": g(best_v, "libero_spatial_swap") - g(best_v)})
    t = pd.DataFrame(rows)
    t.to_csv(out.replace(".md", ".csv"), index=False)
    fmt = lambda v: "" if isinstance(v, float) and np.isnan(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in t.itertuples(index=False)]
    open(out, "w").write("\n".join(lines) + "\n")
    return t


def stage5(out_dir, stage4="results/stage4"):
    os.makedirs(out_dir, exist_ok=True)
    d = pd.read_csv(os.path.join(stage4, "metrics_by_split.csv"))
    done = []
    for target in TARGETS:
        if (d.target == target).any():
            done.append(score_vs_layer(d, target, os.path.join(out_dir, f"{target}_vs_layer.png")))
            done.append(score_vs_layer(d, target, os.path.join(out_dir, f"{target}_by_phase.png"),
                                       positions=("act7",), cols="phase"))
    done.append(plate_presence(pd.read_csv(os.path.join(stage4, "presence_scores.csv")),
                               os.path.join(out_dir, "plate_presence_vs_layer.png")))
    plate = pd.read_csv(os.path.join(stage4, "plate_test.csv"))
    done.append(plate_probe(plate, os.path.join(out_dir, "plate_probe_vs_layer.png")))
    done.append(plate_probe(plate, os.path.join(out_dir, "plate_probe_mixed_vs_layer.png"), exp="plate_mixed"))
    summary_table(d, os.path.join(out_dir, "summary_table.md"))
    return done + [os.path.join(out_dir, "summary_table.md")]


# ----------------------------------------------------------------------------------------------- detail plots
DIRS = ["+x", "-x", "+y", "-y", "+z", "-z", "stay"]                   # labels.DIRECTIONS
CLASS_NAMES = {"direction": DIRS, "distance": [f"bin {i}" for i in range(8)], "gripper": ["open", "close"],
               "contact": ["no", "yes"]}
SPLIT_SETS = {"held-out Spatial": lambda m: m["suite"] == 0,
              "swap, before grasp": lambda m: (m["suite"] == 1) & ~m["phase_after"],
              "swap, after grasp": lambda m: (m["suite"] == 1) & m["phase_after"]}


def best_points(stage5_dir):
    """{target: (point, pos)} from the Stage 5 summary table (best held-out Spatial score)."""
    t = pd.read_csv(os.path.join(stage5_dir, "summary_table.csv"))
    by_title = {v[0]: k for k, v in TARGETS.items()}
    return {by_title[r["question"]]: tuple(r["best point"].split(" / ")) for _, r in t.iterrows()}


def _meta(stage4):
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in torch.load(os.path.join(stage4, "meta.pt"), weights_only=False).items()}


def _true_labels(q, r, meta):
    if q != "distance":
        return meta[q]
    d, y = np.linalg.norm(meta["next_xyz"], axis=1), np.zeros(len(meta["task"]), np.int64)
    for k, e in enumerate(r["edges"]):                                 # split k tests task k; bins use its edges
        te = meta["task"] == k
        y[te] = np.searchsorted(e, d[te], side="right")
    return y


def _cm_panel(ax, y, p, names, title):
    K = len(names)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, p), 1)
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    fs = 7 if K <= 2 else 6 if K <= 7 else 5.5
    for i in range(K):
        for j in range(K):
            if cm[i].sum() == 0:
                continue
            ax.text(j, i, f"{frac[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=fs,
                    color="white" if frac[i, j] > 0.6 else INK)
    rec = [frac[i, i] for i in range(K) if cm[i].sum()]
    ax.set_xticks(range(K)); ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"{n} (n={cm[i].sum()})" for i, n in enumerate(names)], fontsize=7)
    ax.set_title(f"{title}\nbalanced acc {np.mean(rec):.2f}", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("predicted", color=MUTED, fontsize=8)
    ax.set_ylabel("true", color=MUTED, fontsize=8)


def confusion(preds, meta, best, out_dir):
    """Per question: rows = held-out Spatial / swap episodes of the held-out task, columns = linear / MLP (seed 0)."""
    done = []
    for q in ("direction", "distance", "gripper", "contact"):
        pt, pos = best[q]
        r = torch.load(os.path.join(preds, "actions", f"{pt}_{pos}.pt"), weights_only=False)
        y, names = _true_labels(q, r, meta), CLASS_NAMES[q]
        K = len(names)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9.6) if K > 2 else (8, 6.4))
        for i, (suite, sname) in enumerate(((0, "held-out LIBERO-Spatial task"), (1, "same task, LIBERO-PRO swap"))):
            g = meta["suite"] == suite
            for j, (v, vname) in enumerate((("lin", "linear"), ("mlp", "MLP, seed 0"))):
                _cm_panel(axes[i, j], y[g], r[q][v][0][g].astype(int), names, f"{vname} | {sname}")
        fig.suptitle(f"{TARGETS[q][0]}: confusion matrix at {pt} / {pos}, all 10 held-out tasks pooled\n"
                     "cells: fraction of the true class (row-normalized) and count", x=0.01, ha="left",
                     fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95 if K > 2 else 0.92))
        out = os.path.join(out_dir, f"1_confusion_{q}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        done.append(out)
    return done


def scatter_xy(preds, meta, best, out_dir):
    """Q5 predicted vs true image position (x and y separately) for held-out Spatial / swap before / after grasp."""
    pt, pos = best["target_xy"]
    r = torch.load(os.path.join(preds, "position", f"{pt}_{pos}.pt"), weights_only=False)
    y, O = meta["target_xy"], meta["original_plate_xy"]
    done = []
    for v, vname in (("mlp", "MLP, mean of 3 seeds"), ("lin", "linear")):
        p = r["target_xy"][v].astype(np.float32).mean(0)
        fig, axes = plt.subplots(2, 3, figsize=(13, 9.4))
        for j, (gname, gf) in enumerate(SPLIT_SETS.items()):
            g = gf(meta)
            err = np.linalg.norm(p[g] - y[g], axis=1).mean()
            for i, ax_name in enumerate("xy"):
                ax = axes[i, j]
                ax.scatter(y[g, i], p[g, i], s=3, alpha=0.25, color=C[0], lw=0, rasterized=True)
                ax.plot([0, 256], [0, 256], color=INK, lw=1, ls="--", label="y = x")
                if gname == "swap, after grasp":
                    ax.axhline(O[i], color=C[1], lw=1.2, ls=":", label=f"original plate spot ({ax_name} = {O[i]:.0f})")
                    ax.legend(frameon=False, fontsize=7, loc="upper left")
                mae = np.abs(p[g, i] - y[g, i]).mean()
                rr = np.corrcoef(y[g, i], p[g, i])[0, 1]
                ax.set_title(f"{gname} | {ax_name}" + (f" (n={g.sum()}, 2-D error {err:.1f} px)" if i == 0 else "")
                             + f"\nmean |error| {mae:.1f} px  |  r = {rr:.2f}", fontsize=8.5, color=INK, loc="left")
                ax.set_xlim(0, 256); ax.set_ylim(0, 256); ax.set_aspect("equal"); _style(ax)
                ax.set_xlabel(f"true {ax_name} (px, 256 image)", color=MUTED, fontsize=8)
                ax.set_ylabel(f"predicted {ax_name} (px)", color=MUTED, fontsize=8)
        fig.suptitle(f"Q5 target position (bowl before grasp, plate after): {vname} probe at {pt} / {pos}, "
                     "held-out tasks, pooled over queries", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)
        out = os.path.join(out_dir, f"2_target_scatter_{v}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        done.append(out)
    return done


def training_curves(curves_dir, best, out_dir):
    """MLP train / validation loss and score per epoch, every split (thin) and the median over splits (thick)."""
    done = []
    for q in ("direction", "distance", "gripper", "contact", "target_xy"):
        pt, pos = best[q]
        exp = "position" if q == "target_xy" else "actions"
        r = torch.load(os.path.join(curves_dir, f"{exp}_{pt}_{pos}.pt"), weights_only=False)
        reg = q == "target_xy"
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
        for i, shuf in enumerate((False, True)):
            for k, calls in enumerate(r["curves"]):
                for sets, h in zip(r["curve_sets"], calls):
                    m = sets.index((q, 0, shuf)) if (q, 0, shuf) in sets else None
                    if m is None:
                        continue
                    ep = np.arange(1, len(h) + 1)
                    a = {key: np.array([e[key][m] for e in h]) for key in h[0]}
                    best_ep = int(np.argmin(a["val_loss"]))
                    for j, (kt, kv) in enumerate((("train_loss", "val_loss"), ("train_score", "val_score"))):
                        ax = axes[i, j]
                        ax.plot(ep, a[kt], color=C[0], lw=1, alpha=0.6, label="training" if k == 0 else None)
                        ax.plot(ep, a[kv], color=C[1], lw=1, alpha=0.6, label="validation" if k == 0 else None)
                        ax.plot(best_ep + 1, a[kv][best_ep], "o", color=C[1], ms=4, mec="white", mew=0.8,
                                label="kept epoch (lowest val loss)" if k == 0 else None)
            lab = "shuffled labels (seed 0)" if shuf else "real labels (seed 0)"
            axes[i, 0].set_title(f"loss | {lab}", fontsize=9, color=INK, loc="left")
            axes[i, 1].set_title(f"{'mean pixel error' if reg else 'balanced accuracy'} | {lab}", fontsize=9,
                                 color=INK, loc="left")
            axes[i, 0].set_ylabel("MSE (standardized px)" if reg else "class-balanced cross-entropy", color=MUTED,
                                  fontsize=8)
            axes[i, 1].set_ylabel("px (lower is better)" if reg else "balanced accuracy", color=MUTED, fontsize=8)
            if not reg:
                axes[i, 1].set_ylim(0, 1.02)
                axes[i, 1].axhline(TARGETS[q][1], color=MUTED, lw=1, ls=":")
        for ax in axes.flat:
            _style(ax)
        for ax in axes[1]:
            ax.set_xlabel("epoch", color=MUTED)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{TARGETS[q][0]}: MLP training curves at {pt} / {pos}\none line per split (10 task-held-out "
                     "splits; validation = the next task)", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = os.path.join(out_dir, f"3_curves_{q}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        done.append(out)
    return done


def layers_seeds(d, best, out_dir):
    """Score vs layer at each question's best token position, real vs shuffled, error bars = std over the 3 seeds."""
    qs = ("direction", "distance", "gripper", "contact", "target_xy")
    fig, axes = plt.subplots(len(qs), 2, figsize=(11, 3.1 * len(qs)), sharex=True)
    x = np.arange(len(PTS))
    for i, q in enumerate(qs):
        pos = best[q][1]
        reg = q == "target_xy"
        for j, suite in enumerate(SUITE_NAMES):
            ax = axes[i, j]
            z = d[(d.target == q) & (d.pos == pos) & (d.suite == suite) & (d.phase == "all")]
            for v, col, ls, lab in VARIANTS + ([("task_id", MUTED, ":", "task-ID control")] if reg else []):
                zz = z[z.variant == v].set_index("point").reindex(PTS)
                ax.errorbar(x + {"lin": -0.12, "lin_shuf": -0.12, "mlp": 0.12, "mlp_shuf": 0.12}.get(v, 0),
                            zz["mean"], yerr=zz["std_seeds"], color=col, ls=ls, lw=1.6, marker="o" if ls == "-" else
                            None, ms=3.5, capsize=2.5, elinewidth=1, label=lab)
            if not reg:
                ax.axhline(TARGETS[q][1], color=MUTED, lw=1, ls=":", label="chance")
                ax.set_ylim(0, 1.02)
            ax.set_title(f"{TARGETS[q][0]}\n{SUITE_NAMES[suite]} | {POS[pos]}", fontsize=8.5, color=INK, loc="left")
            ax.set_ylabel("px error (lower is better)" if reg else "balanced accuracy", color=MUTED, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(PTS, fontsize=8); _style(ax)
            if i == 0 and j == 0 or reg and j == 0:
                ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Score vs layer, real vs shuffled labels. Error bars: std over the 3 seeds of the score averaged "
                 "over the 10 held-out tasks.\nThe linear probe with real labels is deterministic (no seed spread).",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(out_dir, "4_score_vs_layer_seeds.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return [out]


def presence_heatmap(scores, out_dir, points=("L4", "L16", "L28")):
    """Mean P(plate present) per held-out task (rows) x spot token (columns) at early / middle / late layers."""
    rows = [("libero_spatial_swap", t, s) for t, s in ((0, "A"), (4, "A"), (5, "A"), (8, "A"), (7, "C"), (9, "C"),
                                                        (6, "B"))] + \
           [("libero_spatial", t, "O") for t in (0, 4, 5, 6, 7, 8, 9)]
    spots = ["A", "B", "C", "O"]
    done = []
    for v, vname in (("mlp", "MLP"), ("lin", "linear")):
        fig, axes = plt.subplots(1, len(points), figsize=(13, 7.4), sharey=True)
        for ax, pt in zip(axes, points):
            z = scores[(scores.variant == v) & (scores.point == pt)].set_index(["suite", "task"])
            M = np.array([[z.loc[(su, t), sp] for sp in spots] for su, t, _ in rows])
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i, (su, t, true) in enumerate(rows):
                for j, sp in enumerate(spots):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else INK)
                ax.add_patch(plt.Rectangle((spots.index(true) - 0.5, i - 0.5), 1, 1, fill=False, ec=C[1], lw=2))
            ax.axhline(6.5, color=INK, lw=1)
            ax.set_xticks(range(4)); ax.set_xticklabels([f"spot {s}" for s in spots], fontsize=8)
            ax.set_title({"L4": "early", "L16": "middle", "L28": "late"}.get(pt, "") + f" ({pt})", fontsize=9,
                         color=INK, loc="left")
            ax.tick_params(length=0)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([f"{'swap' if su.endswith('swap') else 'Spatial'} task {t} (plate at {s})"
                                 for su, t, s in rows], fontsize=8)
        fig.suptitle(f"Plate presence, {vname} probe: mean P(plate present) of each spot's patch token, per held-out "
                     "task\norange box = where the plate actually is; spot B has no training positives when task 6 "
                     "is held out", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out = os.path.join(out_dir, f"5_presence_heatmap_{v}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        done.append(out)
    return done


def detail(out_dir, preds, stage4="results/stage4", stage5_dir="results/stage5"):
    """The five detail plot sets, plus one PDF with all of them."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(out_dir, exist_ok=True)
    meta, best = _meta(stage4), best_points(stage5_dir)
    pngs = confusion(preds, meta, best, out_dir) + scatter_xy(preds, meta, best, out_dir) + \
        training_curves(os.path.join(stage4, "curves"), best, out_dir) + \
        layers_seeds(pd.read_csv(os.path.join(stage4, "metrics_by_split.csv")), best, out_dir) + \
        presence_heatmap(pd.read_csv(os.path.join(stage4, "presence_scores.csv")), out_dir)
    pdf = os.path.join(out_dir, "stage4_detail_plots.pdf")
    with PdfPages(pdf) as pp:
        for p in pngs:
            im = plt.imread(p)
            h, w = im.shape[:2]
            fig = plt.figure(figsize=(w / 150, h / 150))
            fig.figimage(im, resize=False)
            pp.savefig(fig, dpi=150); plt.close(fig)
    return pngs + [pdf]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["grasp-traces", "projection", "stage5", "detail"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", default="results/stage4", help="detail: folder with actions/ and position/ predictions")
    a = ap.parse_args()
    if a.what == "detail":
        print("wrote", *detail(a.out, a.preds), sep="\n  ")
    elif a.what == "stage5":
        print("wrote", *stage5(a.out), sep="\n  ")
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        print("wrote", grasp_traces(a.paths, a.out) if a.what == "grasp-traces" else projection(a.paths[0], a.out))
