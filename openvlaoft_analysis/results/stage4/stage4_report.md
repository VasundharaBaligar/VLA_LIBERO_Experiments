# Stage 4 results: OpenVLA-OFT probing (Q1–Q4 and the plate test; Q5 still running)

**Setup.** Frozen OpenVLA-OFT (LIBERO-Spatial checkpoint).
- Saved hidden states: 10 hook points (L0, L4 … L28, L31, final norm) × 3 token positions:
  - act56: mean over the 56 action positions;
  - act7: mean over the first 7;
  - last: the `</s>` token.
- Splits: 10 task-held-out splits. Split k tests on task k, validates on the next task and trains on the other 8, so every task is tested once.
- Probes, each run with real and shuffled labels, 3 seeds:
  - linear: logistic regression or ridge, L2 picked on validation;
  - MLP: 4096→256→ReLU→dropout→out, early stopping on validation.
- Shuffled controls permute both the training and the validation labels. The first run used real validation labels for the control's tuning, which leaked real signal into it; that run was discarded.
- Scores: mean ± std across splits. "Swap" means the LIBERO-PRO episodes of the held-out task.
- "Best layer" means the layer with the highest held-out score, so it is chosen descriptively.

## Q1–Q4: action, gripper, contact (trained on LIBERO-Spatial)

| Question (chance) | Best point | Spatial MLP / linear | Shuffled | Selectivity (MLP) | Swap MLP | Drop to swap | Swap before / after grasp |
|---|---|---|---|---|---|---|---|
| Direction, stay < 0.10 (0.14) | L31, act7 | 0.87 ± 0.08 / 0.83 | 0.13 | 0.74 | 0.75 ± 0.11 | 0.12 | 0.79 / 0.65 |
| Direction, stay < 0.30 (0.14) | norm, act7 | 0.91 ± 0.06 / 0.87 | 0.17 | 0.74 | 0.86 ± 0.08 | 0.05 | 0.87 / 0.83 |
| Distance, 8 bins (0.125) | norm, act7 | 0.52 ± 0.04 / 0.45 | 0.12 | 0.40 | 0.43 ± 0.05 | 0.09 | 0.46 / 0.38 |
| Gripper close (0.5) | L31, act7 | 1.00 / 1.00 | 0.53 | 0.47 | 1.00 | 0.00 | 1.00 / 1.00 |
| Contact (0.5) | L16, act7 | 0.96 ± 0.02 / 0.96 | 0.46 | 0.50 | 0.82 ± 0.11 | 0.14 | 0.83 / 0.83 |

**Layer trends (act7):**
- **Direction** rises steadily with depth: 0.49 at L0, 0.73 at L16, 0.86–0.87 at L28–L31. The last token peaks mid-network (about 0.68 at L20) and then declines.
- **Distance** rises the same way, from 0.27 to 0.52.
- **Gripper** is already 0.94–0.96 at L0. The proprio token carries the current gripper state, and the next command usually matches it.
- **Contact** is flat at about 0.95 at every layer.
- **Linear vs MLP:** the MLP adds at most +0.04–0.07, and only in late layers for direction and distance.

**By phase:**
- In Spatial, scores are similar before and after the grasp.
- Contact after the grasp is 0.74 ± 0.21. That's because there are few no-contact queries after a grasp.
- In swap episodes, direction drops after the grasp (0.65 vs 0.79 before).

**Success vs failure** (pooled over held-out tasks, MLP at the best point):
- Spatial failures score lower than successes: direction 0.73 vs 0.88, contact 0.75 vs 0.97. Only 252 queries from 9 failed episodes.
- The swap successes (all task 5) score higher on direction than the swap failures (0.80 vs 0.74).

**Class counts:**
- With stay < 0.10, −x and −y are rare in Spatial: 103 and 208 queries in total.
- Some Spatial test splits have 0–1 examples of them, so those splits average balanced accuracy over fewer classes.
- Full per-split, per-phase counts: `class_counts.csv`.

## The plate test: does the model see the moved plate, and does it act on it?

**Data.** LIBERO-Spatial puts the plate at the same spot in every task: O = (59, 187) px. After the swap the plate is at:
- A in tasks 0, 2, 4, 5, 8;
- B in task 6;
- C in tasks 7, 9.

Main tasks: 0, 4, 5, 7, 8, 9 (A and C), with task 6 (B) reported separately.

### 1. Where the model *sees* the plate (patch tokens)

- **Data:** 49 rerun episodes (28 swap, 21 Spatial).
- **Probe:** one presence probe shared across the 4 spot tokens. Labels: plate within 17 px of the spot.
- **Score:** each cell is the probability the plate is present. MLP probe, shown as early / middle / late = L4 / L16 / L28.

| Held-out swap task (plate at) | A | B | C | O |
|---|---|---|---|---|
| 0 (A) | **1.00 / 1.00 / 1.00** | 0.00 / 0.00 / 0.00 | 0.24 / 0.12 / 0.06 | 0.08 / 0.06 / 0.02 |
| 4 (A) | **1.00 / 1.00 / 1.00** | 0.17 / 0.21 / 0.22 | 0.28 / 0.27 / 0.27 | 0.14 / 0.16 / 0.19 |
| 5 (A) | **0.91 / 0.90 / 0.85** | 0.04 / 0.02 / 0.06 | 0.04 / 0.08 / 0.12 | 0.25 / 0.19 / 0.28 |
| 8 (A) | **1.00 / 1.00 / 1.00** | 0.15 / 0.25 / 0.29 | 0.38 / 0.23 / 0.30 | 0.01 / 0.01 / 0.10 |
| 7 (C) | 0.00 / 0.00 / 0.00 | 0.21 / 0.18 / 0.20 | **0.41 / 0.41 / 0.41** | 0.10 / 0.04 / 0.24 |
| 9 (C) | 0.00 / 0.01 / 0.06 | 0.15 / 0.29 / 0.47 | **0.36 / 0.38 / 0.35** | 0.00 / 0.00 / 0.03 |
| 6 (B) | 0.01 / 0.02 / 0.07 | **0.01 / 0.02 / 0.03** | 0.38 / 0.61 / 0.61 | 0.02 / 0.13 / 0.35 |
| Spatial, all tasks (O) | ≤ 0.16 | ≤ 0.29 | ≤ 0.47 | **0.81–1.00** |

**What it shows:**
- **Spot A** scores ~1.0 only in tasks 0, 4, 5 and 8, and ≤ 0.07 in the swap tasks 6, 7 and 9. So the probe detects the plate itself, not a "swap scene".
- **Spot O** correctly reads "no plate" in swap episodes and "plate" in Spatial episodes.
- **Spot C** is detected only weakly in tasks 7 and 9 (0.35–0.41), and it also fires falsely in task 6.
- **Spot B** is never detected. When task 6 is held out, the probe has no plate-at-B examples to learn from (B has 0 positives in splits 2 and 3; see `presence_counts.csv`). So task 6 can't be tested this way.
- The linear probe gives the same picture.
- Shuffled controls sit around 0.3–0.6 everywhere, with no task pattern.

**Early to late layers:** "sees the plate at its new spot" does **not** fade.
- A detection is 1.00 from L0 through the final norm.
- In late layers it blurs a little: swap O scores rise (task 6: 0.02 → 0.35) and Spatial O scores dip (0.95 → 0.80–0.90).

**Reference checks:**
- **Random patch token:** predicting its own patch location gives 24–29 px error at L0–L16, vs 95–102 px for its shuffled control. Image tokens clearly carry their own location, which is why the "token at the plate" version can't be trusted.
- **Token at the plate:** predicting the plate's position gives 24–33 px, no better than the random patch locating itself.
- Location readability fades with depth: the random-patch error rises to 38–45 px at L28 / final norm.
- **Mean over all patches:** it locates the plate only weakly (47–58 px, vs 55–59 for the control).

### 2. Where the action positions *point* (plate-position probe, trained on swap training tasks only)

Mean error to the plate's current position, act7 (px, 256 frame):

| Spot group | Linear (L0 → norm) | MLP (L0 → norm) | Shuffled control |
|---|---|---|---|
| A (tasks 0, 4, 5, 8) | 25–38 (best 25 at L28–L31) | 23–30 (best 23 at L0, L28, L31) | 29–35 |
| C (tasks 7, 9) | 56–64 | 58–68 | 57–63 |
| B (task 6) | 81–85 | 82–85 | 84–89 |
| Tasks 1, 2, 3 (separate) | 43–50 | 43–51 | 48–54 |

**Rule:** a layer "encodes the current plate" only if it beats the control by at least 5 px in **every** held-out task of the group.
- **No layer or position meets the rule.** Act56, act7 and last were all checked.
- **A** gets at best 3 of 4 tasks; **C** gets 0–1 of 2.
- In task 6 the real probe is 1–5 px better at a few layers, but it is still 80+ px off.

So the plate's current position can't be read from the action positions any better than the training-average control does. Most of the "A" predictions come from A being the most common spot in training.

**Two artifacts to watch for:**
- **Task 6:** 97–100% of predictions land closer to the *original* spot. But the shuffled control does the same: the training average sits nearer O than B. This is not evidence of memory.
- **Spatial + swap training:** adding Spatial data biases the probe toward O. In group A, the share of predictions closer to the original goes from 1–8% (swap-only) to 13–30% (mixed). That's the size of the Spatial bias.

**Placement probe** (where the bowl ends up): 16–20 px error, the same as its control (18–20). In failures the bowl almost always ends at O, so the average already predicts it.

### 3. Side by side, same held-out swap episodes (L16; MLP presence probe vs MLP plate_swap probe at act7)

| Task | Plate now | Presence probe sees | Action probe points to | Action probe error to now / to original (px) | Bowl actually ends at |
|---|---|---|---|---|---|
| 0 | A | A 100% | A 100% | 13 / 58 | O 100% |
| 4 | A | A 100% | C 65%, A 35% | 46 / 60 | O 100% |
| 5 | A | A 90%, O 10% | A 90% | 16 / 53 | O 52%, A 48% (the 19 successes) |
| 8 | A | A 100% | A 94% | 25 / 75 | O 100% |
| 7 | C | C 61%, B 32% | A 59%, C 25% | 56 / 68 | O 75%, B 25% |
| 9 | C | C 49%, B 46% | B 46%, A 44% | 60 / 81 | O 100% |
| 6 | B | C 85% (B untestable) | A 50%, C 50% | 85 / 48 | O 100% |

Early (L4) and late (L28) layers give the same picture (`presence_side_by_side.csv`).

**Rerun check:** all 49 rerun episodes reproduce Stage 3 (same length and outcome; action chunks match to ≤ 0.016, median 0).

## Plain-English summary

- **Actions, gripper, contact:** the model's action positions encode the next move's direction well (0.87 balanced accuracy, chance 0.14), its size moderately (0.52, chance 0.125), and gripper and contact almost perfectly. The direction and distance signals build up layer by layer and peak in the last layers. In swap scenes these readouts get worse, mostly after the grasp.
- **The moved plate:** the model *sees* it. Image tokens at the plate's new spot (A) say "plate here" and the original spot says "no plate", at every layer, with no fading. But the action positions do **not** carry the plate's new location any better than a shuffled-label control. In 75–100% of failed episodes (100% in five of the seven held-out tasks) the robot puts the bowl at the original spot.
- **Our reading:** the plate's new location is visible in the image tokens but isn't used by the action computation. The policy goes to the memorized spot.
- **Limits:**
  - only 3 moved-plate spots exist, and spot B (task 6) can't be tested;
  - spot C is detected only weakly;
  - the patch-check set is small (402 queries);
  - "not readable" from the action positions means not readable by these linear and MLP probes, not proof the information is absent.
- **Still pending:** Q5 (target position, trained on Spatial + swap, with the task-ID control) is still running; results will be added.
