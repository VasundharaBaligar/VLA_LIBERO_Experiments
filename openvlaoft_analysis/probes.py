"""Stage 4: linear and MLP probes on the saved OpenVLA-OFT hidden states (nothing here touches the VLA itself).

For every hook point (L0..L31, final norm) x token position (act56, act7, last) and every task split, fits
  linear: multinomial logistic regression (classification) or ridge (regression), L2 tuned on the validation task
  MLP:    4096 -> 256 -> ReLU -> dropout -> out, Adam, early stopping on the validation task
each with real and shuffled training labels, 3 seeds. Test predictions for every query are saved (each task is the
test task of exactly one split), and metrics are computed from them by `--summarize`.

Usage:
  python probes.py --exp actions                 # Q1-Q4 (+ direction at the 0.30 stay threshold)
  python probes.py --exp position                # Q5 target position (trained on Spatial + swap)
  python probes.py --summarize                   # tables from the saved predictions
Results: results/stage4/meta.pt (labels per query), results/stage4/<exp>/<point>_<pos>.pt (resumable).
"""
import argparse, glob, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import labels as L

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
PC = CFG["probes"]
OUT = os.path.join(HERE, PC["out_dir"])
SUITES = ["libero_spatial", "libero_spatial_swap"]
POS_NAMES = ["act56", "act7", "last"]
KMAX = 8
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


# ----------------------------------------------------------------------------------------------- data
def load_all():
    """All queries of both suites: hidden (N, 3 positions, n_points, 4096) bf16 + per-query labels / metadata."""
    hid, m = [], {k: [] for k in ("suite", "task", "episode", "success", "mismatch", "q", "phase_after", "next_xyz",
                                   "gripper", "contact", "target_xy", "plate_xy", "bowl_xy", "placement_xy")}
    points = None
    for s, suite in enumerate(SUITES):
        for p in sorted(glob.glob(os.path.join(HERE, CFG["collection"]["out_dir"], suite, "task*_ep*.pt"))):
            e = torch.load(p, weights_only=True, mmap=True)
            n = e["n_queries"]
            if n == 0:
                continue
            assert points in (None, e["hidden_points"]), p
            points = e["hidden_points"]
            hid.append(e["hidden"])
            for k, v in (("suite", s), ("task", e["task_id"]), ("episode", e["episode"]), ("success", e["success"]),
                         ("mismatch", bool(e["wording_mismatch"]))):
                m[k].append(np.full(n, v))
            m["q"].append(np.arange(n))
            m["placement_xy"].append(np.repeat(e["bowl_xy"][-1:].numpy(), n, 0))   # where the bowl ended up
            for k, src in (("phase_after", "phase_after"), ("next_xyz", "next_xyz"), ("gripper", "gripper_close_next"),
                           ("contact", "contact"), ("target_xy", "target_xy"), ("plate_xy", "plate_xy"),
                           ("bowl_xy", "bowl_xy")):
                m[k].append(e[src].numpy())
    meta = {k: np.concatenate(v) for k, v in m.items()}
    lab = CFG["labels"]
    meta["direction"] = np.array([L.direction_label(v, lab["direction_stay_threshold"]) for v in meta["next_xyz"]])
    meta["direction_030"] = np.array([L.direction_label(v, lab["direction_stay_threshold_sensitivity"])
                                      for v in meta["next_xyz"]])
    meta["gripper"] = meta["gripper"].astype(np.int64)
    meta["contact"] = meta["contact"].astype(np.int64)
    meta["points"] = points
    first = (meta["suite"] == 0) & (meta["q"] == 0)
    meta["original_plate_xy"] = meta["plate_xy"][first].mean(0)            # the plate's spot in LIBERO-Spatial
    return torch.cat(hid), meta


def split_tasks(k, n_per_task=None, min_val=200):
    """Split k: test task k, validation task (k + 1) % 10, training = the other 8. If n_per_task (training-pool
    queries per task) is given, the validation task is the next one with at least min_val of them (the after-grasp
    plate experiments: swap task 3 has none and task 1 has 57)."""
    test, val = k, (k + 1) % 10
    while n_per_task is not None and n_per_task[val] < min_val:
        val = (val + 1) % 10
    return [t for t in range(10) if t not in (test, val)], val, test


def distance_labels(meta, train_idx):
    """8 quantile bins of ||next xyz||; edges from this split's training queries."""
    d = np.linalg.norm(meta["next_xyz"], axis=1)
    edges = np.quantile(d[train_idx], np.linspace(0, 1, CFG["labels"]["distance_bins"] + 1)[1:-1])
    return np.searchsorted(edges, d, side="right"), edges


# ----------------------------------------------------------------------------------------------- metrics
def balanced_accuracy(y, p):
    """Mean recall over the classes present in y (NaN if y is empty)."""
    cls = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cls])) if len(cls) else float("nan")


def class_weights(y, K):
    """'balanced' weights n / (n_present * count); classes absent from y get 0."""
    cnt = torch.bincount(y, minlength=K).float()
    w = torch.where(cnt > 0, len(y) / ((cnt > 0).sum() * cnt.clamp(min=1)), torch.zeros_like(cnt))
    return w


# ----------------------------------------------------------------------------------------------- linear probes
def fit_logistic(Xtr, Ytr, Ks, Xva, Yva, l2s, max_iter):
    """Multinomial logistic regression with class-balanced weights and an L2 penalty. Ytr / Yva: (S, n) label sets
    with Ks[s] classes. All (label set, l2) problems are independent; they are fit jointly by one full-batch L-BFGS,
    which converges to the same optima. Returns (W, b) of the l2 with the best validation balanced accuracy per set."""
    S, D, A = len(Ytr), Xtr.shape[1], len(l2s)
    M = S * A
    W = torch.zeros(M, D, KMAX, device=DEV, requires_grad=True)
    b = torch.zeros(M, KMAX, device=DEV, requires_grad=True)
    mask = torch.stack([torch.arange(KMAX, device=DEV) >= Ks[m // A] for m in range(M)]).float() * -1e4  # (M, K)
    lam = torch.tensor([l2s[m % A] for m in range(M)], device=DEV)
    Ytr_m = Ytr.repeat_interleave(A, 0)                                                              # (M, n)
    cw = torch.stack([class_weights(Ytr[m // A], KMAX) for m in range(M)])                          # (M, K)
    wy = cw.gather(1, Ytr_m)                                                                         # (M, n)

    def logits(X):
        return (X @ W.permute(1, 0, 2).reshape(D, M * KMAX)).view(-1, M, KMAX).permute(1, 0, 2) + b[:, None] \
            + mask[:, None]

    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        ce = F.cross_entropy(logits(Xtr).reshape(-1, KMAX), Ytr_m.reshape(-1), reduction="none").view(M, -1)
        loss = ((wy * ce).mean(1) + lam / 2 * (W ** 2).sum((1, 2))).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pv = logits(Xva).argmax(-1).cpu().numpy()
    yv = Yva.cpu().numpy()
    best = [max(range(A), key=lambda a: balanced_accuracy(yv[s], pv[s * A + a])) for s in range(S)]
    sel = [s * A + a for s, a in enumerate(best)]
    return W.detach()[sel], b.detach()[sel] + mask[sel], [l2s[a] for a in best]


def fit_ridge(Xtr, Ytr, Xva, Yva, alphas):
    """Ridge regression. Ytr / Yva: (S, n, 2) target sets (pixels). X is standardized (centered), so the intercept
    is the training target mean. Alpha picked per set on validation mean pixel error. Returns (W (S, D, 2), b, alphas)."""
    S, n, _ = Ytr.shape
    Xd = Xtr.double()
    s, V = torch.linalg.eigh(Xd.T @ Xd)
    ym = Ytr.double().mean(1, keepdim=True)                                                       # (S, 1, 2)
    XtY = Xd.T @ (Ytr.double() - ym).permute(1, 0, 2).reshape(n, S * 2)                            # (D, S*2)
    A = V.T @ XtY
    best, bestW = [None] * S, [None] * S
    for a in alphas:
        Wa = (V @ (A / (s.clamp(min=0) + a)[:, None])).float().view(-1, S, 2).permute(1, 0, 2)    # (S, D, 2)
        pv = torch.einsum("nd,sdk->snk", Xva, Wa) + ym.float()
        err = (pv - Yva).norm(dim=-1).mean(1)
        for i in range(S):
            if best[i] is None or err[i] < best[i][0]:
                best[i], bestW[i] = (float(err[i]), a), Wa[i]
    return torch.stack(bestW), ym.float()[:, 0], [bb[1] for bb in best]


# ----------------------------------------------------------------------------------------------- MLP probes
def fit_mlp(Xtr, Ytr, Xva, Yva, seeds, kind, Ks=None, hist=None):
    """S independent MLPs (4096 -> H -> ReLU -> dropout -> out) trained side by side with batched matmuls.
    Model s has its own init and batch order (seeds[s]) and its own training labels Ytr[s]; early stopping on the
    validation loss (class-balanced CE or MSE on standardized targets); the best-epoch weights are kept.
    kind: 'cls' (Ytr (S, n) int, Ks classes) or 'reg' (Ytr (S, n, 2) float).
    hist: if a list, one dict per epoch is appended with each model's mean training loss (dropout on) and validation
    loss, and its training and validation score (balanced accuracy, or mean pixel error for 'reg'), dropout off."""
    c = PC["mlp"]
    S, (n, D), H = len(seeds), Xtr.shape, c["hidden"]
    out = KMAX if kind == "cls" else 2
    params = []
    for s, sd in enumerate(seeds):
        g = torch.Generator(device=DEV).manual_seed(int(sd))
        u = lambda shape, fan: (torch.rand(shape, generator=g, device=DEV) * 2 - 1) / fan ** 0.5   # nn.Linear init
        params.append([u((D, H), D), u((H,), D), u((H, out), H), u((out,), H)])
    W1, b1, W2, b2 = [torch.stack([p[i] for p in params]).requires_grad_() for i in range(4)]
    if kind == "cls":
        mask = torch.stack([torch.arange(KMAX, device=DEV) >= k for k in Ks]).float() * -1e4
        cw = torch.stack([class_weights(Ytr[s], KMAX) for s in range(S)])
    else:
        ym, ys = Ytr.mean(1, keepdim=True), Ytr.std(1, keepdim=True)
        Ytr, Yva_n = (Ytr - ym) / ys, (Yva - ym) / ys

    def fwd(Xb, train):                                                                  # Xb (S, B, D)
        h = F.dropout(torch.relu(torch.bmm(Xb, W1) + b1[:, None]), c["dropout"], training=train)
        o = torch.bmm(h, W2) + b2[:, None]
        return o + mask[:, None] if kind == "cls" else o

    def loss_per_model(o, y):
        if kind == "cls":
            ce = F.cross_entropy(o.reshape(-1, KMAX), y.reshape(-1), reduction="none").view(S, -1)
            return (cw.gather(1, y) * ce).mean(1)
        return ((o - y) ** 2).mean((1, 2))

    def score_per_model(o, y):
        if kind == "reg":
            return ((o - y) * ys).norm(dim=-1).mean(1)
        oh = F.one_hot(y, KMAX).float()
        cnt, hit = oh.sum(1), (oh * (o.argmax(-1) == y)[..., None]).sum(1)
        return (hit / cnt.clamp(min=1)).sum(1) / (cnt > 0).sum(1)

    opt = torch.optim.Adam([W1, b1, W2, b2], lr=c["lr"])
    g = torch.Generator(device=DEV).manual_seed(int(seeds[0]) + 12345)
    best = torch.full((S,), float("inf"), device=DEV)
    keep = [p.detach().clone() for p in (W1, b1, W2, b2)]
    since = torch.zeros(S, dtype=torch.long, device=DEV)
    Xva_b = Xva[None].expand(S, -1, -1)
    yva = Yva if kind == "cls" else Yva_n
    for ep in range(c["max_epochs"]):
        perm = torch.argsort(torch.rand(S, n, generator=g, device=DEV), dim=1)
        tl = torch.zeros(S, device=DEV)
        for i in range(0, n, c["batch"]):
            idx = perm[:, i:i + c["batch"]]
            yb = Ytr.gather(1, idx) if kind == "cls" else Ytr.gather(1, idx[..., None].expand(-1, -1, 2))
            opt.zero_grad()
            lb = loss_per_model(fwd(Xtr[idx], True), yb)
            lb.sum().backward()
            opt.step()
            tl += lb.detach() * idx.shape[1]
        with torch.no_grad():
            ova = fwd(Xva_b, False)
            vl = loss_per_model(ova, yva)
            if hist is not None:
                hist.append({k: v.cpu().numpy() for k, v in (
                    ("train_loss", tl / n), ("val_loss", vl),
                    ("train_score", score_per_model(fwd(Xtr[None].expand(S, -1, -1), False), Ytr)),
                    ("val_score", score_per_model(ova, yva)))})
            better = vl < best
            best = torch.where(better, vl, best)
            since = torch.where(better, torch.zeros_like(since), since + 1)
            for k, p in zip(keep, (W1, b1, W2, b2)):
                k[better] = p.detach()[better]
        if bool((since >= c["patience"]).all()):
            break
    W1, b1, W2, b2 = keep

    def predict(X, proba=False):
        with torch.no_grad():
            o = torch.bmm(torch.relu(torch.bmm(X[None].expand(S, -1, -1), W1) + b1[:, None]), W2) + b2[:, None]
            if kind == "cls":
                return torch.softmax(o + mask[:, None], -1) if proba else (o + mask[:, None]).argmax(-1)
            return o * ys + ym
    return predict, ep + 1


# ----------------------------------------------------------------------------------------------- experiments
def standardize(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    return (X - mu) / sd


VAL_SHIFT = 1000                     # the shuffled control's validation labels use permutation seed 1000 + seed


def permuted(y, seed):
    g = np.random.default_rng(seed)
    return y[g.permutation(len(y))]


def run_feature(exp, X, meta, log, curves=False):
    """All splits of one experiment on one feature set X (N, D) on the GPU. Returns test predictions per query.
    curves=True also keeps the MLPs' per-epoch history: res['curves'][split] = list of fit_mlp hist lists (one per
    fit_mlp call) and res['curve_sets'] = the (target, seed, shuffled) of each model in those calls."""
    seeds = PC["seeds"]
    N = len(X)
    ec = PC["experiments"][exp]
    suites = [SUITES.index(s) for s in ec["train_suites"]]
    targets = ec["targets"]
    res = dict(hparams=[], epochs=[], edges=[], splits=[], curves=[], curve_sets=[])
    cls = exp == "actions"
    shape = (N,) if cls else (N, 2)
    dtype = np.int8 if cls else np.float16                         # pixels; float16 keeps 0.125 px steps
    for t in targets:
        res[t] = {"lin": np.zeros((1,) + shape, dtype), "lin_shuf": np.zeros((len(seeds),) + shape, dtype),
                  "mlp": np.zeros((len(seeds),) + shape, dtype), "mlp_shuf": np.zeros((len(seeds),) + shape, dtype)}
    in_suites = np.isin(meta["suite"], suites)
    n_per_task = None
    if ec.get("train_phase") == "after":                                 # plate experiments: after-grasp states only
        in_suites &= meta["phase_after"]
        n_per_task = np.bincount(meta["task"][in_suites], minlength=10)
    for k in range(PC["n_splits"]):
        tr_tasks, val_task, test_task = split_tasks(k, n_per_task)
        res["splits"].append((tr_tasks, val_task, test_task))
        tr = np.where(in_suites & np.isin(meta["task"], tr_tasks))[0]
        va = np.where(in_suites & (meta["task"] == val_task))[0]
        te = np.where(meta["task"] == test_task)[0]                          # both suites
        Xs = standardize(X, torch.from_numpy(tr).to(DEV))
        Xtr, Xva, Xte = Xs[tr], Xs[va], Xs[te]
        t0 = time.time()
        hp = dict(split=k)
        res["curves"].append([])
        if cls:
            dist, edges = distance_labels(meta, tr)
            res["edges"].append(edges)
            ys = {"direction": meta["direction"], "direction_030": meta["direction_030"], "distance": dist,
                  "gripper": meta["gripper"], "contact": meta["contact"]}
            Kq = {"direction": 7, "direction_030": 7, "distance": 8, "gripper": 2, "contact": 2}
            # label sets: per target [real, shuffled seed 0, 1, 2]
            sets = [(q, None) for q in targets] + [(q, sd) for q in targets for sd in seeds]
            Ytr = torch.tensor(np.stack([ys[q][tr] if sd is None else permuted(ys[q][tr], sd) for q, sd in sets]),
                               device=DEV)
            Yva = torch.tensor(np.stack([ys[q][va] if sd is None else permuted(ys[q][va], VAL_SHIFT + sd)
                                         for q, sd in sets]), device=DEV)
            Ks = [Kq[q] for q, _ in sets]
            W, b, l2 = fit_logistic(Xtr, Ytr, Ks, Xva, Yva, PC["linear"]["logistic_l2"], PC["linear"]["lbfgs_max_iter"])
            pte = (torch.einsum("nd,sdk->snk", Xte, W) + b[:, None]).argmax(-1).cpu().numpy()
            for i, (q, sd) in enumerate(sets):
                if sd is None:
                    res[q]["lin"][0, te] = pte[i]
                else:
                    res[q]["lin_shuf"][seeds.index(sd), te] = pte[i]
            hp["logistic_l2"] = {f"{q}/{'real' if sd is None else f'shuf{sd}'}": a for (q, sd), a in zip(sets, l2)}
            msets = [(q, sd, sh) for q in targets for sh in (False, True) for sd in seeds]
            Ytr_m = torch.tensor(np.stack([permuted(ys[q][tr], sd) if sh else ys[q][tr] for q, sd, sh in msets]),
                                 device=DEV)
            Yva_m = torch.tensor(np.stack([permuted(ys[q][va], VAL_SHIFT + sd) if sh else ys[q][va]
                                           for q, sd, sh in msets]), device=DEV)
            h = [] if curves else None
            predict, ne = fit_mlp(Xtr, Ytr_m, Xva, Yva_m, [sd for _, sd, _ in msets], "cls", [Kq[q] for q, _, _ in msets],
                                  hist=h)
            res["curves"][k].append(h)
            res["curve_sets"] = [msets]
            pte = predict(Xte).cpu().numpy()
            for i, (q, sd, sh) in enumerate(msets):
                res[q]["mlp_shuf" if sh else "mlp"][seeds.index(sd), te] = pte[i]
        else:
            for q in targets:
                y = meta[q].astype(np.float32)
                sets = [None] + list(seeds)
                Ytr = torch.tensor(np.stack([y[tr] if sd is None else permuted(y[tr], sd) for sd in sets]), device=DEV)
                Yva = torch.tensor(np.stack([y[va] if sd is None else permuted(y[va], VAL_SHIFT + sd) for sd in sets]),
                                   device=DEV)
                W, b, al = fit_ridge(Xtr, Ytr, Xva, Yva, PC["linear"]["ridge_alpha"])
                pte = (torch.einsum("nd,sdk->snk", Xte, W) + b[:, None]).cpu().numpy()
                res[q]["lin"][0, te] = pte[0]
                res[q]["lin_shuf"][:, te] = pte[1:]
                hp[f"ridge_alpha/{q}"] = al
                Ytr_m = torch.tensor(np.stack([y[tr]] * len(seeds) + [permuted(y[tr], sd) for sd in seeds]), device=DEV)
                Yva_m = torch.tensor(np.stack([y[va]] * len(seeds) + [permuted(y[va], VAL_SHIFT + sd) for sd in seeds]),
                                     device=DEV)
                h = [] if curves else None
                predict, ne = fit_mlp(Xtr, Ytr_m, Xva, Yva_m, list(seeds) * 2, "reg", hist=h)
                res["curves"][k].append(h)
                if k == 0:
                    res["curve_sets"].append([(q, sd, sh) for sh in (False, True) for sd in seeds])
                pte = predict(Xte).cpu().numpy()
                res[q]["mlp"][:, te] = pte[:len(seeds)]
                res[q]["mlp_shuf"][:, te] = pte[len(seeds):]
        res["hparams"].append(hp)
        res["epochs"].append(ne)
        log(f"    split {k} (test task {test_task}, val {val_task}): train {len(tr)} val {len(va)} test {len(te)} | "
            f"MLP stopped after {ne} epochs | {time.time() - t0:.0f} s")
    if ec.get("train_phase") == "after":                     # only after-grasp predictions are analyzed: save those
        res["rows"] = np.where(meta["phase_after"])[0]
        for t in targets:
            res[t] = {v: p[:, res["rows"]] for v, p in res[t].items()}
    return res


def task_id_control(meta, exp):
    """Control for position probes: ridge on one-hot(task) + suite flag only. With task-held-out splits the test task's
    one-hot column was never trained, so this predicts the training mean plus a suite offset."""
    suites = [SUITES.index(s) for s in PC["experiments"][exp]["train_suites"]]
    Z = np.concatenate([np.eye(10)[meta["task"]], meta["suite"][:, None], np.ones((len(meta["task"]), 1))], 1)
    out = {}
    for q in PC["experiments"][exp]["targets"]:
        pred = np.zeros((len(Z), 2), np.float32)
        for k in range(PC["n_splits"]):
            tr_tasks, _, test_task = split_tasks(k)
            tr = np.isin(meta["suite"], suites) & np.isin(meta["task"], tr_tasks)
            te = meta["task"] == test_task
            w = np.linalg.lstsq(Z[tr], meta[q][tr], rcond=None)[0]
            pred[te] = Z[te] @ w
        out[q] = pred
    return out


# ----------------------------------------------------------------------------------------------- summary
PHASES = {"all": None, "before": False, "after": True}
OUTCOMES = {"all": None, "success": True, "failure": False}


def _groups(meta, suite, phase=None, outcome=None):
    g = meta["suite"] == suite
    if phase is not None:
        g &= meta["phase_after"] == phase
    if outcome is not None:
        g &= meta["success"] == outcome
    return g


def _score(kind, y, p):
    """cls: balanced accuracy; reg: (mean pixel error, sum sq. error, sum sq. deviation) for pooled R^2."""
    if kind == "cls":
        return balanced_accuracy(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.linalg.norm(p - y, axis=1).mean())


def _r2(y, p):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()) if len(y) > 1 else float("nan")


def summarize():
    """Metric tables from the saved test predictions -> results/stage4/*.csv."""
    import pandas as pd
    meta = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in
            torch.load(os.path.join(OUT, "meta.pt"), weights_only=False).items()}
    dist = np.linalg.norm(meta["next_xyz"], axis=1)
    rows, pooled, counts = [], [], []
    for exp, ec in PC["experiments"].items():
        kind = "cls" if exp == "actions" else "reg"
        ctrl = torch.load(os.path.join(OUT, exp, "task_id_control.pt"), weights_only=False) if kind == "reg" and \
            os.path.exists(os.path.join(OUT, exp, "task_id_control.pt")) else {}
        for path in sorted(glob.glob(os.path.join(OUT, exp, "*_*.pt"))):
            if path.endswith("task_id_control.pt"):
                continue
            pt, pos = os.path.basename(path)[:-3].split("_", 1)
            r = torch.load(path, weights_only=False)
            if "rows" in r:                                           # predictions saved for a subset of queries
                for q in ec["targets"]:
                    for v, p in r[q].items():
                        full = np.full((len(p), len(meta["task"])) + p.shape[2:], np.nan, np.float32)
                        full[:, r["rows"]] = p
                        r[q][v] = full
            for q in ec["targets"]:
                if q == "distance":                                   # labels depend on the split's bin edges
                    y = np.zeros(len(dist), np.int64)
                    for k in range(PC["n_splits"]):
                        te = meta["task"] == split_tasks(k)[2]
                        y[te] = np.searchsorted(r["edges"][k], dist[te], side="right")
                else:
                    y = meta[q]
                variants = dict(r[q])
                if q in ctrl:
                    variants["task_id"] = ctrl[q][None]
                for v, P in variants.items():
                    for s, suite in enumerate(SUITES):
                        for ph, phv in PHASES.items():
                            per_split = []                                 # (splits, seeds)
                            for k in range(PC["n_splits"]):
                                te = _groups(meta, s, phv) & (meta["task"] == split_tasks(k)[2])
                                per_split.append([_score(kind, y[te], p[te]) for p in P])
                            a = np.array(per_split)
                            sm = np.nanmean(a, 1)
                            rows.append(dict(exp=exp, target=q, point=pt, pos=pos, variant=v, suite=suite, phase=ph,
                                             metric="bal_acc" if kind == "cls" else "px_err",
                                             mean=np.nanmean(sm), std_splits=np.nanstd(sm),
                                             std_seeds=np.nanstd(np.nanmean(a, 0)), n_splits=int(np.sum(~np.isnan(sm))),
                                             per_split=";".join(f"{x:.4f}" for x in sm)))
                            for oc, ocv in OUTCOMES.items():         # pooled over splits (every query tested once)
                                g = _groups(meta, s, phv, ocv)
                                if g.sum() == 0:
                                    continue
                                sc = [(_score(kind, y[g], p[g]), _r2(y[g], p[g]) if kind == "reg" else np.nan)
                                      for p in P]
                                pooled.append(dict(exp=exp, target=q, point=pt, pos=pos, variant=v, suite=suite,
                                                   phase=ph, outcome=oc, n=int(g.sum()),
                                                   score=np.mean([x[0] for x in sc]),
                                                   score_std_seeds=np.std([x[0] for x in sc]),
                                                   r2=np.mean([x[1] for x in sc])))
                if kind == "cls" and pt == "L0" and pos == "act56":    # class counts do not depend on the feature
                    for k in range(PC["n_splits"]):
                        for s, suite in enumerate(SUITES):
                            for ph, phv in PHASES.items():
                                te = _groups(meta, s, phv) & (meta["task"] == split_tasks(k)[2])
                                c = np.bincount(y[te], minlength=7 if q.startswith("direction") else
                                                8 if q == "distance" else 2)
                                names = L.DIRECTIONS if q.startswith("direction") else \
                                    [f"bin{i}" for i in range(8)] if q == "distance" else ["no", "yes"]
                                counts.append(dict(target=q, split=k, test_task=split_tasks(k)[2], suite=suite,
                                                   phase=ph, **{n_: int(x) for n_, x in zip(names, c)}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "metrics_by_split.csv"), index=False)
    pd.DataFrame(pooled).to_csv(os.path.join(OUT, "metrics_pooled.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(OUT, "class_counts.csv"), index=False)
    print(f"wrote {len(rows)} + {len(pooled)} + {len(counts)} rows to {OUT}")


# ----------------------------------------------------------------------------------------------- patch check
PCK = CFG["patch_check"]
SPOTS = list(PCK["spots"])


def load_patch_check():
    """Patch-check episodes (collect.py --patch-stride): per kept query the 3 action-position vectors, the spot / plate /
    random tokens and the all-patch mean, with labels. Also matches each query to its Stage 3 episode (same task,
    initial state) to check the rerun reproduces it and to borrow the episode's final bowl position and the
    plate_swap probe predictions for that query."""
    out = {k: [] for k in ("suite", "task", "episode", "q_index", "plate_xy", "placement_xy", "patch_tokens",
                           "repro_chunk_maxdiff", "repro_same_len")}
    act, patch = [], []
    for suite in ("libero_spatial_swap", "libero_spatial"):
        for p in sorted(glob.glob(os.path.join(HERE, PCK["out_dir"], suite, "task*_ep*.pt"))):
            e = torch.load(p, weights_only=True)
            n = e["n_queries"]
            if n == 0:
                continue
            ref = torch.load(os.path.join(HERE, CFG["collection"]["out_dir"], suite, os.path.basename(p)),
                             weights_only=True)
            qi = e["q_index"].numpy()
            same = len(e["trace"]["t"]) == len(ref["trace"]["t"]) and e["success"] == ref["success"]  # same length
            diff = float((e["chunk"] - ref["chunk"][qi]).abs().max()) if ref["n_queries"] > qi.max() else np.inf
            act.append(e["hidden"]); patch.append(e["patch_hidden"])
            for k, v in (("suite", SUITES.index(suite)), ("task", e["task_id"]), ("episode", e["episode"]),
                         ("repro_chunk_maxdiff", diff), ("repro_same_len", same)):
                out[k].append(np.full(n, v))
            out["q_index"].append(qi)
            out["plate_xy"].append(e["plate_xy"].numpy())
            out["placement_xy"].append(np.repeat(ref["bowl_xy"][-1:].numpy(), n, 0))   # Stage 3 episode end
            out["patch_tokens"].append(e["patch_tokens"].numpy())
    meta = {k: np.concatenate(v) for k, v in out.items()}
    return torch.cat(act), torch.cat(patch), meta


def run_presence(log):
    """Plate-presence probe on the fixed-spot patch tokens (shared across spots A, B, C, O), plus the references:
    plate_now token -> plate position, random token -> its own patch position (how readable position is), all-patch
    mean -> plate position. Task-held-out over the 7 main tasks, both suites of a task together."""
    act, patch, m = load_patch_check()
    n, seeds = len(m["task"]), PC["seeds"]
    spot_xy = np.array([PCK["spots"][s] for s in SPOTS], np.float32)
    present = (np.linalg.norm(m["plate_xy"][:, None] - spot_xy[None], axis=-1) < PCK["present_radius_px"]).astype(np.int64)
    tok_xy = lambda tok: np.stack([((tok - 1) % 16 + 0.5) * 14, ((tok - 1) // 16 + 0.5) * 14], -1).astype(np.float32)
    tasks = PCK["tasks"]
    res = dict(meta=m, present=present, splits=[], counts=[])
    points = load_meta_points()
    for pi, pt in enumerate(points):
        R = {"presence": {v: np.zeros((len(seeds) if v != "lin" else 1, n, len(SPOTS)), np.float32)
                          for v in ("lin", "lin_shuf", "mlp", "mlp_shuf")}}
        refs = {"plate_now": (4, m["plate_xy"]), "random": (5, tok_xy(m["patch_tokens"][:, 5])),
                "mean_all": (6, m["plate_xy"])}
        for r in refs:
            R[r] = {v: np.zeros((len(seeds) if v != "lin" else 1, n, 2), np.float32)
                    for v in ("lin", "lin_shuf", "mlp", "mlp_shuf")}
        for k, test_task in enumerate(tasks):
            val_task = tasks[(k + 1) % len(tasks)]
            tr = np.where(~np.isin(m["task"], [test_task, val_task]))[0]
            va, te = np.where(m["task"] == val_task)[0], np.where(m["task"] == test_task)[0]
            if pi == 0:
                res["splits"].append((test_task, val_task))
                for name, idx in (("train", tr), ("val", va), ("test", te)):
                    res["counts"].append(dict(split=k, test_task=test_task, set=name, **{
                        f"{s}_{lab}": int((present[idx, j] == (lab == "present")).sum())
                        for j, s in enumerate(SPOTS) for lab in ("present", "absent")}))
            # presence: one sample per (query, spot)
            X = patch[:, :4, pi].to(DEV, torch.float32)                               # (n, 4, D)
            flat = lambda idx: X[idx].reshape(-1, X.shape[-1])
            ytr, yva = present[tr].reshape(-1), present[va].reshape(-1)
            mu, sd = flat(tr).mean(0), flat(tr).std(0).clamp(min=1e-6)
            Xtr, Xva, Xte = [(flat(i) - mu) / sd for i in (tr, va, te)]
            sets = [None] + list(seeds)
            Ytr = torch.tensor(np.stack([ytr if s is None else permuted(ytr, s) for s in sets]), device=DEV)
            Yva = torch.tensor(np.stack([yva if s is None else permuted(yva, VAL_SHIFT + s) for s in sets]), device=DEV)
            W, b, _ = fit_logistic(Xtr, Ytr, [2] * len(sets), Xva, Yva, PC["linear"]["logistic_l2"],
                                   PC["linear"]["lbfgs_max_iter"])
            pr = torch.softmax(torch.einsum("nd,sdk->snk", Xte, W) + b[:, None], -1)[..., 1].cpu().numpy()
            R["presence"]["lin"][:, te] = pr[:1].reshape(1, len(te), 4)
            R["presence"]["lin_shuf"][:, te] = pr[1:].reshape(len(seeds), len(te), 4)
            Ym = torch.tensor(np.stack([ytr] * len(seeds) + [permuted(ytr, s) for s in seeds]), device=DEV)
            Yvm = torch.tensor(np.stack([yva] * len(seeds) + [permuted(yva, VAL_SHIFT + s) for s in seeds]), device=DEV)
            predict, _ = fit_mlp(Xtr, Ym, Xva, Yvm, list(seeds) * 2, "cls", [2] * 2 * len(seeds))
            pr = predict(Xte, proba=True)[..., 1].cpu().numpy().reshape(2 * len(seeds), len(te), 4)
            R["presence"]["mlp"][:, te], R["presence"]["mlp_shuf"][:, te] = pr[:len(seeds)], pr[len(seeds):]
            # references: regression from one token per query
            for r, (slot, y) in refs.items():
                Xr = patch[:, slot, pi].to(DEV, torch.float32)
                mu, sd = Xr[tr].mean(0), Xr[tr].std(0).clamp(min=1e-6)
                Xtr, Xva, Xte = [(Xr[i] - mu) / sd for i in (tr, va, te)]
                Ytr = torch.tensor(np.stack([y[tr] if s is None else permuted(y[tr], s) for s in sets]), device=DEV)
                Yva = torch.tensor(np.stack([y[va] if s is None else permuted(y[va], VAL_SHIFT + s) for s in sets]),
                                   device=DEV)
                W, b, _ = fit_ridge(Xtr, Ytr, Xva, Yva, PC["linear"]["ridge_alpha"])
                p = (torch.einsum("nd,sdk->snk", Xte, W) + b[:, None]).cpu().numpy()
                R[r]["lin"][:, te], R[r]["lin_shuf"][:, te] = p[:1], p[1:]
                Ym = torch.tensor(np.stack([y[tr]] * len(seeds) + [permuted(y[tr], s) for s in seeds]), device=DEV)
                Yvm = torch.tensor(np.stack([y[va]] * len(seeds) + [permuted(y[va], VAL_SHIFT + s) for s in seeds]),
                                   device=DEV)
                predict, _ = fit_mlp(Xtr, Ym, Xva, Yvm, list(seeds) * 2, "reg")
                p = predict(Xte).cpu().numpy()
                R[r]["mlp"][:, te], R[r]["mlp_shuf"][:, te] = p[:len(seeds)], p[len(seeds):]
        res[pt] = R
        log(f"presence | {pt} done")
    torch.save(res, os.path.join(OUT, "presence", "presence.pt"))


def nearest_spot(xy):
    spot_xy = np.array([PCK["spots"][s] for s in SPOTS], np.float32)
    return np.linalg.norm(xy[..., None, :] - spot_xy, axis=-1).argmin(-1)


def summarize_plate():
    """Plate test (plate_swap / plate_mixed) on after-grasp swap queries of each held-out task -> plate_test.csv.
    Distances (px, 256 frame) of each probe's predictions to the plate now, the original spot and the bowl's final
    position; share of queries closer to the original spot than to the plate now."""
    import pandas as pd
    meta = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in
            torch.load(os.path.join(OUT, "meta.pt"), weights_only=False).items()}
    orig = meta["original_plate_xy"]
    sel = (meta["suite"] == 1) & meta["phase_after"]
    rows = []
    for exp in ("plate_swap", "plate_mixed"):
        for path in sorted(glob.glob(os.path.join(OUT, exp, "*_*.pt"))):
            pt, pos = os.path.basename(path)[:-3].split("_", 1)
            r = torch.load(path, weights_only=False)
            idx = r.get("rows", np.arange(len(meta["task"])))
            for q in ("plate_xy", "placement_xy"):
                for v, P in r[q].items():
                    P = P.astype(np.float32)
                    for t in range(10):
                        g = sel[idx] & (meta["task"][idx] == t)
                        if g.sum() == 0:
                            continue
                        gi = idx[g]
                        d = lambda ref: np.linalg.norm(P[:, g] - ref, axis=-1)           # (seeds, n)
                        dn, do, dp = d(meta["plate_xy"][gi]), d(orig), d(meta["placement_xy"][gi])
                        rows.append(dict(exp=exp, target=q, point=pt, pos=pos, variant=v, task=t, n=int(g.sum()),
                                         to_plate_now=dn.mean(), to_original=do.mean(), to_placement=dp.mean(),
                                         closer_to_original=(do < dn).mean(),
                                         to_plate_now_seed_std=dn.mean(1).std()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "plate_test.csv"), index=False)
    return df


def summarize_presence():
    """Presence probe + references + side-by-side with the action-position plate probes -> presence_*.csv."""
    import pandas as pd
    res = torch.load(os.path.join(OUT, "presence", "presence.pt"), weights_only=False)
    m, present = res["meta"], res["present"]
    meta = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in
            torch.load(os.path.join(OUT, "meta.pt"), weights_only=False).items()}
    key = {(s, t, e, q): i for i, (s, t, e, q) in
           enumerate(zip(meta["suite"], meta["task"], meta["episode"], meta["q"]))}
    s3 = np.array([key.get((s, t, e, q), -1) for s, t, e, q in zip(m["suite"], m["task"], m["episode"], m["q_index"])])
    true_spot = np.where(present.any(1), present.argmax(1), -1)
    scores, side, refs = [], [], []
    for pt in [p for p in res if p.startswith("L") or p == "norm"]:
        R = res[pt]
        for v, P in R["presence"].items():
            pm = P.mean(0)                                                         # (n, 4) seed-mean probability
            seen = pm.argmax(1)
            for suite in (1, 0):
                for t in PCK["tasks"]:
                    g = (m["suite"] == suite) & (m["task"] == t)
                    if g.sum() == 0:
                        continue
                    scores.append(dict(point=pt, variant=v, suite=SUITES[suite], task=t, n=int(g.sum()),
                                       true_spot=SPOTS[true_spot[g][0]] if true_spot[g][0] >= 0 else "none",
                                       **{s: float(pm[g, j].mean()) for j, s in enumerate(SPOTS)},
                                       seen_at_true_spot=float((seen[g] == true_spot[g]).mean()),
                                       seen_at_O=float((seen[g] == SPOTS.index("O")).mean())))
        for r in ("plate_now", "random", "mean_all"):
            y = m["plate_xy"] if r != "random" else np.stack(
                [((m["patch_tokens"][:, 5] - 1) % 16 + 0.5) * 14, ((m["patch_tokens"][:, 5] - 1) // 16 + 0.5) * 14], -1)
            for v, P in R[r].items():
                err = np.linalg.norm(P - y, axis=-1).mean(0)                           # per query, seed-mean
                refs.append(dict(point=pt, reference=r, variant=v, frame="256 px" if r != "random" else "224 px",
                                 mean_err=float(err.mean()), swap_mean_err=float(err[m["suite"] == 1].mean())))
        # side by side on held-out swap episodes: presence (MLP) vs plate_swap probe (same layer) vs bowl end
        pm = R["presence"]["mlp"].mean(0)
        for pos in POS_NAMES:
            f = os.path.join(OUT, "plate_swap", f"{pt}_{pos}.pt")
            if not os.path.exists(f):
                continue
            pr = torch.load(f, weights_only=False)
            rows = pr.get("rows", np.arange(len(meta["task"])))
            inv = np.full(len(meta["task"]), -1); inv[rows] = np.arange(len(rows))
            for v in ("lin", "mlp"):
                P = pr["plate_xy"][v].astype(np.float32).mean(0)
                for t in PCK["tasks"]:
                    g = (m["suite"] == 1) & (m["task"] == t) & (s3 >= 0)
                    g &= inv[np.maximum(s3, 0)] >= 0
                    if g.sum() == 0:
                        continue
                    act_xy = P[inv[s3[g]]]
                    side.append(dict(point=pt, act_pos=pos, act_variant=v, task=t, n=int(g.sum()),
                                     plate_now=SPOTS[true_spot[g][0]] if true_spot[g][0] >= 0 else "none",
                                     presence_sees=pd.Series([SPOTS[i] for i in pm[g].argmax(1)]).value_counts(
                                         normalize=True).round(2).to_dict(),
                                     action_points=pd.Series([SPOTS[i] for i in nearest_spot(act_xy)]).value_counts(
                                         normalize=True).round(2).to_dict(),
                                     act_to_plate_now=float(np.linalg.norm(act_xy - m["plate_xy"][g], axis=1).mean()),
                                     act_to_original=float(np.linalg.norm(act_xy - meta["original_plate_xy"],
                                                                          axis=1).mean()),
                                     bowl_ends=pd.Series([SPOTS[i] for i in nearest_spot(m["placement_xy"][g])]
                                                         ).value_counts(normalize=True).round(2).to_dict()))
    for name, rows in (("scores", scores), ("side_by_side", side), ("references", refs)):
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"presence_{name}.csv"), index=False)
    pd.DataFrame(res["counts"]).to_csv(os.path.join(OUT, "presence_counts.csv"), index=False)
    ep = pd.DataFrame(dict(suite=m["suite"], task=m["task"], episode=m["episode"], same=m["repro_same_len"],
                           diff=m["repro_chunk_maxdiff"])).groupby(["suite", "task", "episode"]).first()
    print(f"rerun reproduction: {int(ep.same.sum())}/{len(ep)} episodes same length and outcome as Stage 3; "
          f"max |action chunk diff| at kept queries: median {ep['diff'].median():.3g}, max {ep['diff'].max():.3g}")


def load_meta_points():
    return torch.load(os.path.join(OUT, "meta.pt"), weights_only=False)["points"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(PC["experiments"]) + ["presence"])
    ap.add_argument("--points", nargs="+", default=None, help="subset of hook points, e.g. L16 norm")
    ap.add_argument("--positions", nargs="+", default=POS_NAMES)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--curves", action="store_true", help="refit the MLPs at the given --points/--positions with "
                    "per-epoch logging -> results/stage4/curves/<exp>_<point>_<pos>.pt")
    a = ap.parse_args()
    if a.summarize:
        summarize()
        if glob.glob(os.path.join(OUT, "plate_swap", "*.pt")):
            summarize_plate()
        if os.path.exists(os.path.join(OUT, "presence", "presence.pt")):
            summarize_presence()
        return
    torch.manual_seed(0); np.random.seed(0)
    logdir = os.path.join(OUT, "curves" if a.curves else a.exp)
    os.makedirs(logdir, exist_ok=True)
    logf = open(os.path.join(logdir, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    if a.exp == "presence":
        return run_presence(log)
    H, meta = load_all()
    log(f"loaded {len(H)} queries in {time.time() - t0:.0f} s; points {meta['points']}")
    if not a.curves:
        torch.save({k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
                   os.path.join(OUT, "meta.pt"))
    if a.exp == "position" and not a.curves:
        torch.save(task_id_control(meta, a.exp), os.path.join(OUT, a.exp, "task_id_control.pt"))
    log(f"config: {yaml.safe_dump(PC, sort_keys=False)}")
    for pt in a.points or meta["points"]:
        for pos in a.positions:
            path = os.path.join(OUT, "curves", f"{a.exp}_{pt}_{pos}.pt") if a.curves else \
                os.path.join(OUT, a.exp, f"{pt}_{pos}.pt")
            if os.path.exists(path):
                continue
            log(f"{a.exp} | {pt} {pos}" + (" | training curves" if a.curves else ""))
            X = H[:, POS_NAMES.index(pos), meta["points"].index(pt)].to(DEV, torch.float32)
            res = run_feature(a.exp, X, meta, log, curves=a.curves)
            torch.save(res, path + ".tmp"); os.replace(path + ".tmp", path)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
