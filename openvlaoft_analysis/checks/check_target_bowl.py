"""Stage 1 check: in every LIBERO-Spatial task, is obj_of_interest[0] the bowl the instruction describes (not the
distractor bowl)? For each task the instruction's landmark is located in the scene, and the target bowl must be the
one nearer to it (3D distance, initial state 0, after the 10 settle steps of run_libero_eval.py).
Usage: python checks/check_target_bowl.py"""
import os

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LANDMARK = {                                                 # instruction phrase -> scene body (or list = midpoint)
    "between the plate and the ramekin": ["plate_1", "glazed_rim_porcelain_ramekin_1"],
    "from table center": "TABLE_CENTER", "in the top drawer of the wooden cabinet": "wooden_cabinet_1",
    "next to the cookie box": "cookies_1", "next to the plate": "plate_1", "next to the ramekin": "glazed_rim_porcelain_ramekin_1",
    "on the cookie box": "cookies_1", "on the ramekin": "glazed_rim_porcelain_ramekin_1", "on the stove": "flat_stove_1",
    "on the wooden cabinet": "wooden_cabinet_1"}


def body_pos(sim, name):
    ids = [i for i in range(sim.model.nbody) if (sim.model.body_id2name(i) or "").startswith(name)]
    return sim.data.body_xpos[ids[0]].copy()


suite = benchmark.get_benchmark_dict()["libero_spatial"]()
for t in range(suite.n_tasks):
    task = suite.get_task(t)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                                         task.bddl_file), camera_heights=128, camera_widths=128)
    env.seed(0); env.reset(); env.set_init_state(suite.get_task_init_states(t)[0])
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    sim, target = env.env.sim, env.env.obj_of_interest[0]
    phrase = next(k for k in LANDMARK if k in task.language)
    lm = LANDMARK[phrase]
    if lm == "TABLE_CENTER":
        b1, b2 = body_pos(sim, "akita_black_bowl_1"), body_pos(sim, "akita_black_bowl_2")
        ref = body_pos(sim, "main_table") if any((sim.model.body_id2name(i) or "").startswith("main_table")
                                                  for i in range(sim.model.nbody)) else np.zeros(3)
        ref = ref.copy(); ref[2] = b1[2]
    else:
        ref = np.mean([body_pos(sim, n) for n in (lm if isinstance(lm, list) else [lm])], axis=0)
    d = {b: float(np.linalg.norm(body_pos(sim, b)[:2] - ref[:2])) for b in ("akita_black_bowl_1", "akita_black_bowl_2")}
    near = min(d, key=d.get)
    if phrase == "in the top drawer of the wooden cabinet":   # BDDL: target In the drawer, distractor On the cabinet top
        z = {b: float(body_pos(sim, b)[2]) for b in d}             # -> the drawer bowl is the lower one
        near = min(z, key=z.get)
        print(f"  (task {t}: drawer vs cabinet top decided by height: bowl_1 z={z['akita_black_bowl_1']:.3f}, "
              f"bowl_2 z={z['akita_black_bowl_2']:.3f})")
    print(f"task {t} | {phrase:40s} | target={target} | xy-dist to landmark: bowl_1 {d['akita_black_bowl_1']:.3f} m, "
          f"bowl_2 {d['akita_black_bowl_2']:.3f} m | target is nearer: {near == target}")
    env.close()
