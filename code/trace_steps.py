# Trace OpenVLA's first N decisions on one LIBERO-10 task, recording every stage of the pipeline:
# camera image -> model input -> 7 action tokens -> bins -> normalized [-1,1] -> real action -> robot motion.
import io, os, math, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL_REPO = "openvla/openvla-7b-finetuned-libero-10"
TASK_ID, INIT, N = 2, 0, 10          # "turn on the stove and put the moka pot on it", starting setup 0
OUT = "/workspace/fa26-vla-cs670/rollouts"
NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

def prep_image(obs):
    img = obs["agentview_image"][::-1, ::-1]
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG"); buf.seek(0)
    im = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)
    s = math.sqrt(0.9); c = int(224 * s); off = (224 - c) // 2
    return im.crop((off, off, off + c, off + c)).resize((224, 224), Image.LANCZOS)

processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(MODEL_REPO, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
stats = vla.norm_stats["libero_10"]["action"]
q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
mask = np.array(stats.get("mask", [True] * 7))

suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(TASK_ID)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[INIT])
for _ in range(10):
    obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])

prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
records = []
for step in range(N):
    model_img = prep_image(obs)
    inputs = processor(prompt, model_img).to("cuda:0", dtype=torch.bfloat16)
    ids = inputs["input_ids"]
    if ids[0, -1] != 29871:                                   # same special token predict_action appends
        ids = torch.cat([ids, torch.tensor([[29871]], device=ids.device)], dim=1)
    with torch.no_grad():
        gen = vla.generate(input_ids=ids, pixel_values=inputs["pixel_values"],
                           attention_mask=torch.ones_like(ids), max_new_tokens=7, do_sample=False)
    tokens = gen[0, -7:].cpu().numpy()
    bins = np.clip(vla.vocab_size - tokens - 1, 0, vla.bin_centers.shape[0] - 1)
    normed = vla.bin_centers[bins]
    action = np.where(mask, 0.5 * (normed + 1) * (q99 - q01) + q01, normed)
    env_action = action.copy(); env_action[-1] = -np.sign(2 * env_action[-1] - 1)

    pos_before = obs["robot0_eef_pos"].copy()
    obs, _, done, _ = env.step(env_action.tolist())
    pos_after = obs["robot0_eef_pos"].copy()
    records.append(dict(step=step, tokens=tokens.tolist(), bins=bins.tolist(), normalized=normed.round(3).tolist(),
                        action=action.round(4).tolist(), env_action=env_action.round(4).tolist(),
                        commanded_cm=(np.clip(env_action[:3], -1, 1) * 5).round(2).tolist(),
                        moved_cm=((pos_after - pos_before) * 100).round(2).tolist(), img=model_img))
env.close()

# ---- print the pipeline for every step ----
print(f"Task: {task.language}\nPrompt: {prompt!r}  ({ids.shape[1]} text tokens)\n")
print("libero_10 action range used for un-normalizing (q01 .. q99):")
for n, lo, hi in zip(NAMES, q01, q99):
    print(f"   {n:8s} {lo:+.3f} .. {hi:+.3f}")
for r in records:
    print(f"\n--- step {r['step']} ---")
    print("  7 output tokens   :", r["tokens"])
    print("  bin (0-255)       :", r["bins"])
    print("  normalized [-1,1] :", r["normalized"])
    print("  real action       :", dict(zip(NAMES, r["action"])))
    print("  sent to simulator :", r["env_action"], "(gripper -1=open, +1=close)")
    print("  commanded move cm :", r["commanded_cm"], "  actually moved cm:", r["moved_cm"])

# ---- figure: the 10 model inputs with the chosen move under each ----
fig, axes = plt.subplots(2, 5, figsize=(18, 9))
for ax, r in zip(axes.flat, records):
    ax.imshow(r["img"]); ax.axis("off")
    a = r["env_action"]
    ax.set_title(f"step {r['step']}  (model input 224x224)", fontsize=10)
    ax.text(0.5, -0.03, f"dx={a[0]:+.2f} dy={a[1]:+.2f} dz={a[2]:+.2f}\n"
                        f"rot=({a[3]:+.2f},{a[4]:+.2f},{a[5]:+.2f})  grip={'close' if a[6] > 0 else 'open'}\n"
                        f"hand moved {np.linalg.norm(r['moved_cm']):.2f} cm",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, family="monospace")
fig.suptitle(f'First {N} OpenVLA decisions: "{task.language}"', fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/first_10_moves.png", dpi=120)
json.dump([{k: v for k, v in r.items() if k != "img"} for r in records],
          open(f"{OUT}/first_10_moves.json", "w"), indent=1)
print(f"\nsaved {OUT}/first_10_moves.png and .json")
