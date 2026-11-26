# Stage 5: OpenVLA-OFT probing study, final summary

**Model and probes.**
- Model: `moojink/openvla-7b-oft-finetuned-libero-spatial`, frozen (eval mode, no gradients).
- Hidden states: 10 hook points (L0, L4 … L28, L31, final norm) × 3 token positions:
  - act56: mean of the 56 action positions;
  - act7: mean of the first 7;
  - last: `</s>`.
- Probes: linear (logistic regression or ridge) and MLP (4096→256→ReLU→dropout), each with shuffled-label controls and 3 seeds.
- Splits: 10 task-held-out splits (every task tested once). The LIBERO-PRO position-swap episodes of the held-out task are evaluated in the same split.
- Error bands in the figures: ±1 std across splits.

## Main finding

**Image tokens encode the moved plate (spot A), but the action positions do not, and the robot places the bowl at the original spot.**

- **Spot A** (the plate's new location in tasks 0, 4, 5 and 8):
  - the A patch token scores ~1.0 "plate present" in exactly those tasks, and ≤ 0.07 in the other swap tasks;
  - the original spot's token reads "no plate";
  - this holds at every layer, from L0 to the final norm.
- **The action-position states**: a probe trained to read the plate's current position from them never beats its shuffled-label control by the required margin (≥ 5 px in every held-out task), at any layer or token position.
- **The robot**: in held-out swap episodes the bowl ends at the original plate spot in 100% of episodes for 5 of the 7 tasks.
- **Scope of the claim: spot A only.**
  - Spot C (tasks 7, 9) is weak: its presence score, about 0.4, is at the shuffled-control level, so C is effectively not detected.
  - Spot B (task 6) can't be tested: when task 6 is held out, no plate-at-B example exists in training.

Figures: `plate_presence_vs_layer.png` (seeing), `plate_probe_vs_layer.png` (acting). The Stage 4 report has the per-task tables and the side-by-side comparison.

## Summary table

Best hook point and position chosen on held-out LIBERO-Spatial scores, so the choice is descriptive. "Swap" is the same probe on the position-swap episodes of the held-out task.

| Question | Metric (chance) | Best point | Linear | MLP | Shuffled | Selectivity | Swap | Normal → swap |
|---|---|---|---|---|---|---|---|---|
| Q1 direction (stay < 0.10) | bal. acc (0.14) | L31 / act7 | 0.83 | 0.87 | 0.13 | 0.75 | 0.75 | −0.13 |
| Q1 direction (stay < 0.30, sensitivity) | bal. acc (0.14) | norm / act7 | 0.87 | 0.91 | 0.17 | 0.74 | 0.86 | −0.04 |
| Q2 distance (8 bins) | bal. acc (0.125) | norm / act7 | 0.45 | 0.52 | 0.12 | 0.40 | 0.43 | −0.09 |
| Q3 gripper close | bal. acc (0.5) | L31 / act7 | 1.00 | 1.00 | 0.53 | 0.47 | 1.00 | 0.00 |
| Q4 contact | bal. acc (0.5) | L16 / act7 | 0.96 | 0.96 | 0.46 | 0.50 | 0.82 | −0.14 |
| Q5 target position | px error, lower is better (task-ID control 59) | L12 / act56 | 35.5 | 31.9 | 61.6 | 29.8 | 45.3 | +13.4 px |

**Q5 by phase** (L12 / act56, MLP vs shuffled vs task-ID control, px):

| | Before grasp (target = bowl) | After grasp (target = plate) |
|---|---|---|
| Spatial | 38 vs 71 / 72 | 24 vs 50 / 44 |
| Swap | 46 vs 71 / 70 | **49 vs 46 / 53**: no better than the controls |

Two things to know about these numbers:
- After the grasp in Spatial the plate never moves, so the 24 px there is easy, and R² isn't meaningful (the target barely varies).
- In swap episodes after the grasp, the target-position probe does not find the moved plate. That matches the main finding.

## Plain-English summary

1. **What the action positions encode.** The hidden states at the action positions encode, ahead of time:
   - the direction of the next move (0.87 balanced accuracy vs 0.14 chance);
   - roughly how far it goes (0.52 vs 0.125);
   - whether the gripper will close (≈ 1.00);
   - whether the gripper touches the bowl (0.96).

   Direction and distance build up through the layers and are clearest in the last ones, in the first-7-action readout. The `</s>` token peaks mid-network and then declines. Gripper and contact are readable from the first layer on: the robot-state (proprio) input token carries the gripper state.
2. **Linear vs MLP.** The MLP adds little, at most +0.04–0.07 in late layers for direction and distance. Most of this information is linearly readable.
3. **Controls.** All controls behave: shuffled labels give chance-level scores at every layer, and selectivity is large for Q1–Q5.
4. **The swap breaks things after the grasp.** When the plate is moved, the readouts get worse, mostly after the grasp: direction drops from 0.79 to 0.65 between before and after the grasp, and target position after the grasp is no better than the controls.
5. **Sees, but doesn't use.** The model sees the moved plate at spot A in its image tokens at every layer. That information doesn't show up in the action positions, and the robot carries the bowl to the memorized original spot. In short, the moved plate is visible in the image tokens but not used for the action.
6. **Caveats.**
   - Only spot A supports the claim: C is at control level and B is untestable.
   - Only three moved-plate locations exist.
   - The patch-token set is small (402 queries).
   - "Not readable" means not readable by these linear and MLP probes; it doesn't prove the information is absent.
   - Rare direction classes (−x, −y) have 0–1 examples in some test splits.
   - Selecting the best layer on test scores is descriptive.

## Figures (this folder)

| File | What it shows |
|---|---|
| `<question>_vs_layer.png` | Score vs layer for each question. Rows: held-out Spatial task / swap episodes. Columns: token position. Lines: linear, MLP, linear-shuffled, MLP-shuffled. Q5 also shows the task-ID control. |
| `<question>_by_phase.png` | The same at act7, split into before vs after grasp. |
| `plate_presence_vs_layer.png` | Presence probe across layers: true spot vs the original spot. |
| `plate_probe_vs_layer.png` | Action-position plate probe vs its control, per spot group (swap-only training). |
| `plate_probe_mixed_vs_layer.png` | The same probe trained on Spatial + swap, showing the bias toward the original spot. |
