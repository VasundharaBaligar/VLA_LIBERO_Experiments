# VLA Generalization Experiments on LIBERO

Experiments on how well vision-language-action (VLA) models generalize in the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) robot-manipulation simulator. We compare three VLAs
(**OpenVLA-7B**, **π0.5** and **GR00T N1.5**) on two questions:

1. **Can a VLA act zero-shot?** Base checkpoints, with no LIBERO training, are run closed-loop in LIBERO.
2. **Does the model still know where objects are?** Attention heatmaps from the object's name are scored against
   exact object masks, at each training stage: the VLM it started from, the base VLA, and the VLA fine-tuned on LIBERO.

We also run the LIBERO-fine-tuned OpenVLA on its own tasks and on a custom scene with a changed object.

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [How to reproduce](#3-how-to-reproduce)
4. [Notes and limitations](#4-notes-and-limitations)

---

## 1. Results at a glance

### Fine-tuned OpenVLA on LIBERO-10

`openvla/openvla-7b-finetuned-libero-10`, 3 tasks × 3 trials (`code/rollout_openvla.py`).

| Task | Successes |
|---|:---:|
| Task 2: turn on the stove and put the moka pot on it | 1 / 3 |
| Task 7: put both the alphabet soup and the cream cheese box in the basket | 2 / 3 |
| Task 9: put the yellow and white mug in the microwave and close it | 1 / 3 |
| **Total** | **4 / 9** |

**Changed-instruction test.** Task 9 with a new sentence for the same scene:

| Variant | How success is checked | Outcome |
|---|---|---|
| "put the white mug in the microwave" | custom BDDL goal (`code/custom_bddl/`) | 0 / 3 succeed |
| "pick the pink mug and place it in the microwave" | instruction only; LIBERO checks the task's original goal | in 1 of 3 trials the robot reached the **original** goal, ignoring the new sentence |

Videos and final frames: `results/rollouts/`.

### Zero-shot base VLAs (no LIBERO training)

Task 0 of each of the 4 LIBERO suites × 2 trials, 220 steps each (`code/zeroshot/libero_client.py`).

| Model | Successes | Mean hand travel |
|---|:---:|:---:|
| OpenVLA-7B base | 0 / 8 | 37.8 cm |
| π0.5 base | 0 / 8 | 78.8 cm |
| GR00T N1.5 base | 0 / 8 | 38.9 cm |

> None of the base models completes a LIBERO task without LIBERO training. They do move the arm, so the failure is
> in the policy, not in the control interface.

### Object-location heatmaps

23 objects across 4 LIBERO scenes. For each object, the model's attention from the object's name onto the image patches
is scored against the exact object mask:

- **Hit:** the highest-attention patch lies on the object.
- **Lift:** attention mass on the object ÷ its share of the image (1.0 = no better than uniform).

| Model | Stage | Hits | Mean lift |
|---|---|:---:|:---:|
| **OpenVLA** | VLM (Prismatic) | **13 / 23** | 3.60× |
| | VLA base | 5 / 23 | 2.50× |
| | fine-tuned LIBERO-Spatial | 1 / 23 | 2.00× |
| | fine-tuned LIBERO-Object | 1 / 23 | 1.88× |
| | fine-tuned LIBERO-Goal | 1 / 23 | 1.89× |
| | fine-tuned LIBERO-10 | 2 / 23 | 2.35× |
| **π0.5** | VLM (PaliGemma) | **10 / 23** | 2.89× |
| | VLA base | 7 / 23 | 5.20× |
| **GR00T N1.5** | VLM stand-in (Eagle2-2B) | **8 / 23** | 2.95× |
| | VLA base | 6 / 23 | 1.39× |
| | fine-tuned (each of the 4 suites) | 6 / 23 | 1.39× |

> For every model, the VLM localizes objects best by hit count. OpenVLA's hits drop at each stage: 13 → 5 → 1–2 after
> LIBERO fine-tuning. GR00T's scores are identical across the base and all fine-tuned checkpoints.

Per-object scores: `results/zeroshot_results/heatmaps/*_heatmap_scores.json`.
Side-by-side figures: `results/zeroshot_results/heatmaps/compare/`.

---

## 2. Repository layout

```
.
├── code/
│   ├── rollout_openvla.py        fine-tuned OpenVLA rollouts on LIBERO-10
│   ├── trace_steps.py            traces OpenVLA's first decisions: image → tokens → bins → action
│   ├── viz_suites.py             one example scene per LIBERO suite
│   ├── custom_bddl/              custom LIBERO scene (changed mug)
│   ├── shims/pytorch3d/          minimal pytorch3d stand-in needed by GR00T
│   └── zeroshot/                 zero-shot rollouts and heatmap experiments
│       ├── make_scenes.py        renders scenes + exact object masks
│       ├── heatmap_*.py          heatmaps per model family; heatmap_common.py scores them
│       ├── compare_heatmaps.py   side-by-side comparison figures
│       ├── libero_client.py      LIBERO rollout loop for any policy
│       ├── policy_*.py           policy servers (OpenVLA, π0.5, GR00T); policy_io.py is the protocol
│       └── run_*.sh              end-to-end drivers per model
├── patches/                      small fixes applied to the LIBERO and Isaac-GR00T checkouts
├── results/
│   ├── rollouts/                 fine-tuned OpenVLA videos, final frames, step trace
│   ├── zeroshot_results/         zero-shot rollouts per model, scenes + masks, heatmaps
│   └── benchmarks/               overview images of LIBERO-Plus, CALVIN and SimplerEnv
├── notebooks/OpenVLA_testing.ipynb
└── logs/                         console output of the runs
```

---

## 3. How to reproduce

### Third-party code

Clone the two repos at the commits we used and apply our patches:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout 8f1084e
git -C LIBERO apply ../patches/LIBERO.patch            # torch.load(..., weights_only=False) for newer PyTorch

git clone https://github.com/NVIDIA/Isaac-GR00T.git && git -C Isaac-GR00T checkout 4af2b62
git -C Isaac-GR00T apply ../patches/Isaac-GR00T.patch  # Eagle backbone loading fixes
```

### Python environments

Each model family runs in its own environment, because their dependencies conflict. Policies run as small
localhost servers (`policy_io.py`), and one LIBERO client drives them all.

| Environment | Used for |
|---|---|
| `env` | LIBERO simulator, OpenVLA (`transformers`) |
| `env_pi` | π0.5 via LeRobot |
| `env_groot` | GR00T N1.5 via Isaac-GR00T |

### Run

The scripts expect this layout: `/workspace/libero_run` holds the code and environments, and results go to
`/workspace/fa26-vla-cs670`. Adjust the paths at the top of each script if yours differ.

| Experiment | Command |
|---|---|
| Fine-tuned OpenVLA rollouts | `python code/rollout_openvla.py` |
| Scenes + object masks | `python code/zeroshot/make_scenes.py` |
| OpenVLA heatmaps (all stages) | `bash code/zeroshot/run_all_openvla_heatmaps.sh` |
| π0.5 heatmaps + zero-shot rollouts | `bash code/zeroshot/run_pi05.sh` |
| GR00T zero-shot rollouts + heatmaps | `bash code/zeroshot/run_groot.sh` |
| OpenVLA zero-shot rollouts | `python code/zeroshot/libero_client.py --policy openvla_base` |

The drivers download one checkpoint at a time and delete it afterwards, so everything fits on a 32 GB disk.

---

## 4. Notes and limitations

- Small samples: 2–3 trials per task and 23 objects in 4 scenes. These are first indications, not benchmark numbers.
- GR00T's own VLM is not public, so `nvidia/Eagle2-2B` stands in for it.
- The π0.5 base checkpoint ships without normalization stats, so the LIBERO dataset's min/max range is used.
- OpenVLA base needs an un-normalization key from one of its pretraining datasets; see `policy_openvla.py`.
- Base models: `openvla/openvla-7b`, `lerobot/pi05_base`, `nvidia/GR00T-N1.5-3B`.
