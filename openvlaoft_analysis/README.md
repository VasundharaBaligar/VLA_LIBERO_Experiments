# Probing OpenVLA-OFT hidden states

What do the frozen hidden states of OpenVLA-OFT (`moojink/openvla-7b-oft-finetuned-libero-spatial`) encode about the next action, the gripper, contact and the target's image position? And when the plate is moved (LIBERO-PRO position swap), does the model still know where it is?

**Main finding:** image tokens encode the moved plate (spot A), but the action positions do not, and the robot places the bowl at the original spot. Spot C is weak and spot B is untestable. Details: `results/stage5/stage5_report.md`.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Every setting: model flags, suites, hook layers and positions, label definitions, grasp rule, probe hyperparameters, splits, seeds, patch-check spots. |
| `collect.py` | Runs the official OpenVLA-OFT LIBERO episode loop with forward hooks and saves hidden states and labels per episode. `--check` verifies the hooks don't change actions. `--patch-stride` saves patch tokens for the plate-presence check. |
| `labels.py` | Label definitions: direction and stay threshold, distance bins, grasp / phase rule, contact, image projection, patch-token index, swap wording flag, safe init-state loader. |
| `probes.py` | Linear and MLP probes with shuffled controls on task-held-out splits (`--exp actions / position / plate_swap / plate_mixed / presence`), and `--summarize` to build the tables. |
| `plots.py` | Stage 2 check figures, and `stage5`: score-vs-layer figures, plate figures and the summary table. |
| `checks/` | One-off checks from Stages 1–3: model and LIBERO inspection, target-bowl check, Stage 2 summary, success rates. |

## Reproduce

```bash
# Stage 3: collect (resumable; stops cleanly below 0.5 GB free disk)
python collect.py --suite libero_spatial --n-episodes 50 --video-dir data/videos
python collect.py --suite libero_spatial_swap --n-episodes 50 --video-dir data/videos
# Stage 4: probes
python probes.py --exp actions        # Q1-Q4, plus direction at stay < 0.30
python probes.py --exp position       # Q5 (trained on Spatial + swap)
python probes.py --exp plate_swap     # plate test, swap-only training
python probes.py --exp plate_mixed    # same, Spatial + swap (shows the bias)
python collect.py --suite libero_spatial_swap --tasks 0 4 5 6 7 8 9 --episodes 0 1 2 3 --patch-stride 2 --out data/patch_check
python collect.py --suite libero_spatial --tasks 0 4 5 6 7 8 9 --episodes 0 1 2 --patch-stride 1 --out data/patch_check
python probes.py --exp presence
python probes.py --summarize
# Stage 5
python plots.py stage5 --out results/stage5
```

## Environment

- Python 3.10, torch 2.7.1+cu128, `moojink/transformers-openvla-oft` (4.40.1).
- openvla-oft at commit e4287e9 (`third_party/openvla-oft`).
- LIBERO, plus the LIBERO-PRO swap files from `hf datasets/zhouxueyang/LIBERO-Pro`.
- GPU: RTX PRO 4500 (32 GB).

## Key choices (all in config.yaml)

- **Official evaluation settings:** seed 7, `env.seed(0)`, 10 settle steps, 220 max steps, 8-step action chunks, center crop.
- **Per-episode seed:** 7 + 1000·task + episode.
- **Hidden states:** saved at every model query (once per 8 steps) for L0, L4 … L28, L31 and the final norm, at act56 / act7 / last.
- **Labels:**
  - Direction: largest-magnitude xyz component of the next action; "stay" if every component is below 0.10 (sensitivity check at 0.30).
  - Distance: 8 quantile bins, with edges from each split's training data.
  - Gripper: next commanded gripper action.
  - Contact: robosuite contact check between the gripper and the bowl.
  - Target: bowl before the grasp, plate after.
  - Grasp rule: contact + close command + bowl ≥ 2 cm above rest, latched.
- **Splits:** 10 task-held-out splits (test task k, validation task k+1, train on the other 8).
- **Probes:**
  - Linear: L2 picked on validation.
  - MLP: early stopping on validation.
  - Shuffled controls: permute both training and validation labels.
- **Plate test:** after-grasp states only. Presence probe on fixed patch tokens at spots A, B, C and O, so that a token's built-in position signal can't fake the answer.
