| question | metric | chance | best point | linear | MLP | best probe | shuffled (same probe) | selectivity | task-ID control | swap | normal -> swap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 direction of the next action (stay < 0.10) | balanced acc | 0.14 | L31 / act7 | 0.83 | 0.87 | MLP | 0.13 | 0.75 |  | 0.75 | -0.13 |
| Q1 direction, sensitivity check (stay < 0.30) | balanced acc | 0.14 | norm / act7 | 0.87 | 0.91 | MLP | 0.17 | 0.74 |  | 0.86 | -0.04 |
| Q2 distance of the next action (8 quantile bins) | balanced acc | 0.12 | norm / act7 | 0.45 | 0.52 | MLP | 0.12 | 0.40 |  | 0.43 | -0.09 |
| Q3 next gripper command = close | balanced acc | 0.50 | L31 / act7 | 1.00 | 1.00 | MLP | 0.53 | 0.47 |  | 1.00 | -0.00 |
| Q4 gripper-bowl contact | balanced acc | 0.50 | L16 / act7 | 0.96 | 0.96 | MLP | 0.46 | 0.50 |  | 0.82 | -0.14 |
| Q5 target position in the image (bowl before grasp, plate after) | px error |  | L12 / act56 | 35.53 | 31.85 | MLP | 61.62 | 29.77 | 59.01 | 45.28 | 13.43 |
