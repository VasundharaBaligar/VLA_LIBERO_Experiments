"""Stage 1 check: where each probe label comes from in LIBERO, verified on one LIBERO-Spatial episode start.
Target object (BDDL obj_of_interest), its 3D position, gripper state, gripper-object contact, and the projection of
the 3D position into the policy's main camera image (agentview, rotated 180 degrees as in openvla-oft).
The projection is checked against the object's ground-truth segmentation mask.
Usage: python inspect_libero.py [--task 0]"""
import argparse, os

import numpy as np
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

H = W = 256                                                  # env_img_res in openvla-oft run_libero_eval.py


def object_mask(sim, name):
    seg = sim.render(width=W, height=H, camera_name="agentview", segmentation=True)[..., 1][::-1, ::-1]
    names = np.array([sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)] + [""])
    img = names[np.where(seg >= 0, seg, len(names) - 1)]
    return np.char.startswith(img, name + "_") | (img == name)


def project(sim, p):
    """World point -> (x, y) pixel in the image the policy sees (agentview rotated 180 degrees)."""
    M = get_camera_transform_matrix(sim, "agentview", H, W)
    row, col = project_points_from_world_to_camera(p[None], M, H, W)[0]   # robosuite returns (row, col)
    # robosuite's projection is for the raw rendered frame, which MuJoCo returns upside down; openvla-oft then
    # rotates it 180 degrees ([::-1, ::-1]). Net effect on the raw (row, col) is a horizontal flip only.
    return float(W - 1 - col), float(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    args = ap.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                                         task.bddl_file), camera_heights=H, camera_widths=W)
    env.seed(0); env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[0])
    for _ in range(10):                                      # num_steps_wait in run_libero_eval.py
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    inner, sim = env.env, env.env.sim
    print("task:", task.language)
    print("obj_of_interest (BDDL):", inner.obj_of_interest)
    target = inner.obj_of_interest[0]
    print("objects:", list(inner.objects_dict), "| fixtures:", list(inner.fixtures_dict))
    pos = obs[f"{target}_pos"]
    print(f"{target}_pos (obs):", pos.round(4))
    print("gripper qpos (2 fingers):", obs["robot0_gripper_qpos"].round(4))
    gripper = inner.robots[0].gripper
    print("contact gripper-target:", inner.check_contact(gripper, inner.objects_dict[target]))
    print("grasp (robosuite _check_grasp):", inner._check_grasp(gripper, inner.objects_dict[target]))
    M = get_camera_transform_matrix(sim, "agentview", H, W)
    print("agentview world->pixel matrix:\n", M.round(3))
    x, y = project(sim, pos)
    mask = object_mask(sim, target)
    ys, xs = np.nonzero(mask)
    inside = bool(mask[int(round(y)), int(round(x))])
    print(f"projected target: ({x:.1f}, {y:.1f}) | mask centroid: ({xs.mean():.1f}, {ys.mean():.1f}) | "
          f"projected point inside mask: {inside}")
    img = Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).resize((512, 512))
    d = ImageDraw.Draw(img)
    d.ellipse([2 * x - 6, 2 * y - 6, 2 * x + 6, 2 * y + 6], outline="red", width=3)
    d.ellipse([2 * xs.mean() - 4, 2 * ys.mean() - 4, 2 * xs.mean() + 4, 2 * ys.mean() + 4], fill="cyan")
    out = os.path.join(os.path.dirname(__file__), "stage1_projection_check.png")
    img.save(out); print("wrote", out)
    env.close()


if __name__ == "__main__":
    main()
