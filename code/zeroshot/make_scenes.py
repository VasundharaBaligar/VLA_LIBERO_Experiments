# Render one scene per LIBERO suite and save the front image plus an exact pixel mask for every visible object,
# so every model's attention heatmap can be scored against ground truth.
import os, json
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT = "/workspace/fa26-vla-cs670/zeroshot_results/scenes"
SCENES = [  # (suite, task id, [(name used in the prompt, LIBERO instance name(s) joined by "+"), ...])
    ("libero_spatial", 0, [("black bowl", "akita_black_bowl_1+akita_black_bowl_2"), ("cookie box", "cookies_1"),
                           ("ramekin", "glazed_rim_porcelain_ramekin_1"), ("plate", "plate_1"),
                           ("wooden cabinet", "wooden_cabinet_1"), ("stove", "flat_stove_1")]),
    ("libero_object",  0, [("alphabet soup can", "alphabet_soup_1"), ("basket", "basket_1"),
                           ("salad dressing bottle", "salad_dressing_1"), ("cream cheese box", "cream_cheese_1"),
                           ("milk carton", "milk_1"), ("tomato sauce can", "tomato_sauce_1"), ("butter", "butter_1")]),
    ("libero_goal",    0, [("black bowl", "akita_black_bowl_1"), ("cream cheese box", "cream_cheese_1"),
                           ("wine bottle", "wine_bottle_1"), ("plate", "plate_1"), ("wooden cabinet", "wooden_cabinet_1"),
                           ("stove", "flat_stove_1"), ("wine rack", "wine_rack_1")]),
    ("libero_10",      2, [("frying pan", "chefmate_8_frypan_1"), ("moka pot", "moka_pot_1"), ("stove", "flat_stove_1")]),
]
os.makedirs(OUT, exist_ok=True)
meta = []
for suite_name, task_id, objects in SCENES:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    sim = env.env.sim
    front = obs["agentview_image"][::-1, ::-1]
    seg = sim.render(width=256, height=256, camera_name="agentview", segmentation=True)  # (H, W, 2): objtype, objid
    geom_ids = seg[..., 1][::-1, ::-1]
    geom_names = np.array([sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)] + [""])
    names_img = geom_names[np.where(geom_ids >= 0, geom_ids, len(geom_names) - 1)]
    key = f"{suite_name}_task{task_id}"
    Image.fromarray(front).save(f"{OUT}/{key}.png")
    entry = dict(key=key, suite=suite_name, task=task.language, image=f"{key}.png", objects=[])
    for prompt_name, inst in objects:
        mask = np.zeros(names_img.shape, bool)
        for i in inst.split("+"):
            mask |= np.char.startswith(names_img, i + "_") | (names_img == i)
        assert mask.sum() > 0, (inst, sorted({n.split("_")[0] for n in geom_names}))
        np.save(f"{OUT}/{key}_{inst.split('+')[0]}_mask.npy", mask)
        ys, xs = np.nonzero(mask)
        entry["objects"].append(dict(name=prompt_name, instance=inst, mask=f"{key}_{inst.split('+')[0]}_mask.npy",
                                     pixels=int(mask.sum()), center=[float(xs.mean()), float(ys.mean())]))
        print(key, prompt_name, "mask pixels:", int(mask.sum()))
    meta.append(entry)
    env.close()
json.dump(meta, open(f"{OUT}/scenes.json", "w"), indent=1)
