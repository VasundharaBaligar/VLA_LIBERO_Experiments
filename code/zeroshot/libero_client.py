# Zero-shot LIBERO rollouts: task 0 of each suite x N tries, any policy (in-process OpenVLA or a policy server).
# Usage: python libero_client.py --policy openvla_base | --port 5555 --name pi05_base
import argparse, json, os, sys
import numpy as np
import imageio
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

sys.path.insert(0, os.path.dirname(__file__))
from policy_io import RemotePolicy

SUITES = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # OpenVLA's step limits
WAIT_STEPS = 10

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default=None, help="'openvla_base' to run in-process")
ap.add_argument("--port", type=int, default=None, help="port of a policy server")
ap.add_argument("--name", default=None)
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--suites", default=",".join(SUITES))
ap.add_argument("--out", default="/workspace/fa26-vla-cs670/zeroshot_results")
args = ap.parse_args()
name = args.name or args.policy
out_dir = os.path.join(args.out, name); os.makedirs(out_dir, exist_ok=True)

if args.policy == "openvla_base":
    from policy_openvla import OpenVLABase
    policy = OpenVLABase()
else:
    policy = RemotePolicy(args.port)

def get_obs(obs):
    return {
        "front": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),          # upright 256x256
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]]).astype(np.float32),   # 8-d, as in the LIBERO dataset
    }

def caption(frame, lines):
    im = Image.fromarray(frame).resize((384, 384), Image.LANCZOS)
    canvas = Image.new("RGB", (384, 384 + 16 * len(lines) + 6), "white"); canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        d.text((6, 388 + 16 * i), t, fill="black")
    return np.asarray(canvas)

results = []
for suite_name in args.suites.split(","):
    max_steps = SUITES[suite_name]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(args.task)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.seed(0)
    init_states = suite.get_task_init_states(args.task)
    for trial in range(args.trials):
        env.reset(); obs = env.set_init_state(init_states[trial])
        policy.reset(task.language)
        frames, done, start = [], False, None
        for t in range(max_steps + WAIT_STEPS):
            if t < WAIT_STEPS:
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1]); continue
            if start is None:
                start = obs["robot0_eef_pos"].copy()
            o = get_obs(obs)
            action = np.asarray(policy.act({**o, "instruction": task.language}), dtype=np.float64)
            obs, _, done, _ = env.step(action.tolist())
            frames.append(caption(o["front"], [f"{name} | {suite_name} task {args.task}", task.language[:60],
                                               f"step {t - WAIT_STEPS:3d}  {'SUCCESS' if done else 'running'}"]))
            if done:
                break
        steps = t - WAIT_STEPS + 1
        travel = float(np.linalg.norm(obs["robot0_eef_pos"] - start))
        tag = "success" if done else "fail"
        path = os.path.join(out_dir, f"{suite_name}_task{args.task}_trial{trial}_{tag}.mp4")
        imageio.mimwrite(path, frames, fps=20, quality=8)          # 20 fps = real robot speed
        Image.fromarray(frames[-1]).save(path.replace(".mp4", "_final.png"))
        print(f"[{name}] {suite_name} task {args.task} trial {trial}: {tag} in {steps} steps, "
              f"hand ended {travel * 100:.1f} cm from start -> {path}", flush=True)
        results.append(dict(model=name, suite=suite_name, task=task.language, trial=trial,
                            success=bool(done), steps=steps, hand_travel_cm=round(travel * 100, 1)))
    env.close()

json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
print(f"\n{name}: {sum(r['success'] for r in results)}/{len(results)} successes")
