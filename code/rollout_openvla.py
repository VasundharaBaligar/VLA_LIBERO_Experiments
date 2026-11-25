# Closed-loop OpenVLA rollouts in the LIBERO-10 simulator: the model sees the camera image,
# predicts an action, the simulator executes it, and we repeat until the task is done.
# Follows openvla/experiments/robot/libero/run_libero_eval.py.
import io, os, sys, math
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL_REPO = "openvla/openvla-7b-finetuned-libero-10"
KEYWORDS   = ["mug in the microwave", "moka pot on it", "alphabet soup and the cream cheese"]
MAX_STEPS  = 520                         # LIBERO-10 episode budget used by OpenVLA
WAIT_STEPS = 10                          # let objects settle before acting
TRIALS     = int(os.environ.get("TRIALS", 3))   # init states to try per task
TASK_IDS   = os.environ.get("TASK_IDS")             # e.g. "9" to run only that task (overrides KEYWORDS)
INSTRUCTION = os.environ.get("INSTRUCTION")         # custom instruction replacing the task's own sentence
BDDL_FILE  = os.environ.get("BDDL_FILE")            # custom task file (same scene, new goal) so LIBERO checks the new goal
TRACK      = [tuple(c.split()) for c in os.environ.get("TRACK", "").split(";") if c.strip()]  # extra predicates to report
OUT_DIR    = os.environ.get("OUT_DIR", "/workspace/fa26-vla-cs670/rollouts")
os.makedirs(OUT_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to("cuda:0")
unnorm_key = "libero_10"

def prep_image(obs):
    # LIBERO renders upside down; OpenVLA was trained on the flipped image, JPEG-compressed, 224x224
    img = obs["agentview_image"][::-1, ::-1]
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG"); buf.seek(0)
    im = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)
    # 90% area center crop, matching the random-crop augmentation used in fine-tuning
    s = math.sqrt(0.9); cw = ch = int(224 * s); off = (224 - cw) // 2
    return img, im.crop((off, off, off + cw, off + ch)).resize((224, 224), Image.LANCZOS)

def to_env_action(a):
    a = a.copy()
    a[-1] = 2 * a[-1] - 1                # gripper [0,1] -> [-1,1]
    a[-1] = -np.sign(a[-1])              # binarize and flip: OpenVLA uses 1=open, LIBERO uses -1=open
    return a

def caption(frame, lines):
    im = Image.fromarray(frame).resize((512, 512), Image.LANCZOS)
    canvas = Image.new("RGB", (512, 512 + 18 * len(lines) + 8), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        d.text((8, 516 + 18 * i), t, fill="black")
    return np.asarray(canvas)

suite = benchmark.get_benchmark_dict()["libero_10"]()
summary = []
for task_id in range(suite.n_tasks):
    task = suite.get_task(task_id)
    if TASK_IDS is not None:
        if str(task_id) not in TASK_IDS.split(","):
            continue
    elif not any(k in task.language for k in KEYWORDS):
        continue
    bddl = BDDL_FILE or os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    init_states = suite.get_task_init_states(task_id)
    language = INSTRUCTION or task.language
    prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
    slug = language.replace(" ", "_")[:60]

    for trial in range(TRIALS):
        env.reset()
        obs = env.set_init_state(init_states[trial])
        frames, done, t = [], False, 0
        while t < MAX_STEPS + WAIT_STEPS:
            if t < WAIT_STEPS:
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1]); t += 1
                continue
            raw, model_img = prep_image(obs)
            inputs = processor(prompt, model_img).to("cuda:0", dtype=torch.bfloat16)
            with torch.no_grad():
                action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            obs, _, done, _ = env.step(to_env_action(np.asarray(action, dtype=np.float64)).tolist())
            # `done` is LIBERO's check for the task's ORIGINAL goal, even with a custom instruction
            status = ("original goal reached" if INSTRUCTION and not BDDL_FILE else "SUCCESS - task complete") if done else "running"
            frames.append(caption(raw, [language, f"step {t - WAIT_STEPS:3d}   {status}"]))
            t += 1
            if done:
                break
        if done:                              # hold on the finished scene for a second
            frames += [frames[-1]] * 30
        tag = ("original_goal_reached" if INSTRUCTION and not BDDL_FILE else "success") if done \
              else ("ended" if INSTRUCTION and not BDDL_FILE else "fail")
        path = os.path.join(OUT_DIR, f"task{task_id}_{slug}_trial{trial}_{tag}.mp4")
        imageio.mimwrite(path, frames, fps=30, quality=8)
        Image.fromarray(frames[-1]).save(path.replace(".mp4", "_final.png"))
        print(f"[task {task_id}] trial {trial}: {tag} in {t - WAIT_STEPS} steps -> {path}", flush=True)
        for pred in TRACK:                    # e.g. which mug ended up in the microwave
            print(f"      {' '.join(pred)}: {bool(env.env._eval_predicate(pred))}", flush=True)
        summary.append((language, trial, done, t - WAIT_STEPS))
    env.close()

print("\nSUMMARY")
for lang, trial, ok, steps in summary:
    print(f"  {'OK  ' if ok else 'FAIL'}  trial {trial}  {steps:3d} steps  {lang}")
