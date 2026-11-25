# One example scene per LIBERO suite: front camera (what OpenVLA sees) next to the wrist camera.
import os, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITES = [("libero_spatial", "Spatial: same objects, different layouts"),
          ("libero_object",  "Object: same layout, different objects"),
          ("libero_goal",    "Goal: same scene, different goals"),
          ("libero_10",      "LIBERO-10: long, multi-step tasks")]
OUT = "/workspace/fa26-vla-cs670/rollouts/libero_suites_overview.png"

fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))
for col, (name, blurb) in enumerate(SUITES):
    suite = benchmark.get_benchmark_dict()[name]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    front = obs["agentview_image"][::-1, ::-1]
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    env.close()
    axes[0, col].imshow(front); axes[0, col].axis("off")
    axes[0, col].set_title(f"{blurb}\n({suite.n_tasks} tasks)\n\nFRONT camera  <- OpenVLA's only input", fontsize=11)
    axes[1, col].imshow(wrist); axes[1, col].axis("off")
    axes[1, col].set_title("WRIST camera (not used by OpenVLA)", fontsize=11)
    axes[1, col].text(0.5, -0.08, "\n".join(textwrap.wrap(f'Task: "{task.language}"', 42)),
                      transform=axes[1, col].transAxes, ha="center", va="top", fontsize=11)
fig.suptitle("LIBERO: one example scene from each suite (task 0, starting setup 0)", fontsize=14)
plt.tight_layout(); plt.savefig(OUT, dpi=130); print(OUT)
