# Object-location test for pi0.5: its VLA (pi05_base's PaliGemma part) vs its VLM (google/paligemma-3b-pt-224).
# Same inputs for both: attention heatmap from the object word + PaliGemma's own "detect <object>" box output.
# Usage (env_pi): python heatmap_paligemma.py vla | vlm
import os, re, sys, json
import numpy as np, torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

sys.path.insert(0, os.path.dirname(__file__))
from heatmap_common import load_scenes, score, save_grids, save_summary

STAGE = sys.argv[1]
OUT = "/workspace/fa26-vla-cs670/zeroshot_results/heatmaps"
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

if STAGE == "vla":
    from pi05_common import load_pi05
    policy, _ = load_pi05()
    model = policy.model.paligemma_with_expert.paligemma       # PaliGemma subclass inside pi0.5
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224", torch_dtype=torch.bfloat16).to("cuda")
model.eval()
model.config._attn_implementation = "eager"
model.model.language_model.config._attn_implementation = "eager"

captured = []
for layer in model.model.language_model.layers:
    layer.self_attn.register_forward_hook(lambda m, i, o: captured.append(o[1]))

def inputs_for(text, img):
    enc = processor(text=text, images=Image.fromarray(img), return_tensors="pt")
    return {k: (v.to("cuda", torch.bfloat16) if v.is_floating_point() else v.to("cuda")) for k, v in enc.items()}

def parse_box(text):
    locs = [int(x) for x in re.findall(r"<loc(\d{4})>", text)[:4]]
    if len(locs) < 4:
        return None
    y0, x0, y1, x1 = [v / 1024 * 256 for v in locs]       # PaliGemma boxes are on a 0-1023 grid
    return [x0, y0, x1, y1]

rows, boxes = [], []
for scene in load_scenes():
    for obj in scene["objects"]:
        # ---- 1. attention heatmap ----
        question = f"Where is the {obj['name']}?"
        enc = inputs_for(question, scene["img"])
        ids = enc["input_ids"][0].tolist()
        img_pos = [i for i, t in enumerate(ids) if t == model.config.image_token_index]
        name_ids = processor.tokenizer(" " + obj["name"], add_special_tokens=False)["input_ids"]
        q = next(list(range(s, s + len(name_ids))) for s in range(len(ids)) if ids[s:s + len(name_ids)] == name_ids)
        captured.clear()
        with torch.no_grad():
            model(**enc)
        att = torch.stack([a[0][:, q][:, :, img_pos].float().mean(dim=(0, 1)) for a in captured])   # (layers, 256)
        grid = att.mean(0).reshape(16, 16).cpu().numpy()
        hit, lift, peak = score(grid, obj["mask_arr"])
        # ---- 2. explicit location: "detect <object>" -> bounding box ----
        enc = inputs_for(f"detect {obj['name']}", scene["img"])
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=20, do_sample=False)
        answer = processor.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        box = parse_box(answer)
        box_hit = False
        if box:
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            box_hit = bool(obj["mask_arr"][min(max(cy, 0), 255), min(max(cx, 0), 255)])
        boxes.append(dict(scene=scene["suite"], object=obj["name"], answer=answer.strip(), box=box, box_center_on_object=box_hit))
        rows.append(dict(scene=scene["suite"], key=scene["key"], label=obj["name"], img=scene["img"], mask=obj["mask_arr"],
                         grid=grid, hit=hit, lift=lift, peak=peak, box=box))
        print(scene["suite"], obj["name"], f"| heatmap hit {hit} lift {lift:.2f} | detect -> {answer.strip()!r} on-object {box_hit}")

save_grids(rows, "pi05", STAGE, OUT)          # per-scene comparison figures are drawn by compare_heatmaps.py
save_summary(rows, "pi05", STAGE, OUT)
json.dump(boxes, open(f"{OUT}/pi05_{STAGE}_detect.json", "w"), indent=1)
print(f"detect: {sum(b['box_center_on_object'] for b in boxes)}/{len(boxes)} boxes centered on the right object")
