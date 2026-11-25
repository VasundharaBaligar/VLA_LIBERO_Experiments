# List every LIBERO object/fixture visible in each heatmap scene, with its pixel count in the front camera.
import os, re
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

for suite_name, task_id in [("libero_spatial", 0), ("libero_object", 0), ("libero_goal", 0), ("libero_10", 2)]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset(); env.set_init_state(suite.get_task_init_states(task_id)[0])
    sim = env.env.sim
    insts = list(env.env.objects_dict) + list(env.env.fixtures_dict)
    seg = sim.render(width=256, height=256, camera_name="agentview", segmentation=True)[..., 1]
    names = np.array([sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)] + [""])
    img = names[np.where(seg >= 0, seg, len(names) - 1)]
    print(f"\n{suite_name} task{task_id}: {task.language}")
    for inst in insts:
        px = int((np.char.startswith(img, inst + "_") | (img == inst)).sum())
        print(f"   {inst:35s} {px:6d} px")
    env.close()
