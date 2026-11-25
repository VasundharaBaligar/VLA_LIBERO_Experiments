# Object-location heatmaps for GR00T N1.5: its VLA backbone (Eagle 2.5, first 12 Qwen3 layers kept by GR00T),
# GR00T fine-tuned on LIBERO, and a public stand-in for its VLM (nvidia/Eagle2-2B; GR00T's own Eagle VLM is not public).
# Usage (env_groot): python heatmap_groot.py vla | vlm | ft_<suite>
import os, re, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from heatmap_common import load_scenes, score, save_grids, save_summary

STAGE = sys.argv[1]
OUT = "/workspace/fa26-vla-cs670/zeroshot_results/heatmaps"
FT_REPOS = {"libero_spatial": "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
            "libero_object": "youliangtan/gr00t-n1.5-libero-object-posttrain",
            "libero_goal": "youliangtan/gr00t-n1.5-libero-goal-posttrain",
            "libero_10": "youliangtan/gr00t-n1.5-libero-long-posttrain"}

if STAGE == "vlm":
    from transformers import AutoConfig, AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True, use_fast=True)
    cfg = AutoConfig.from_pretrained("nvidia/Eagle2-2B", trust_remote_code=True)
    for c in (cfg, cfg.text_config, cfg.vision_config):     # its config forces flash-attn, which isn't installed
        c._attn_implementation, c._attn_implementation_autoset = "eager", True
    vlm = AutoModel.from_pretrained("nvidia/Eagle2-2B", config=cfg, trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to("cuda").eval()
    vlm.vision_model.config._attn_implementation = "sdpa"   # its __init__ switches the vision tower back to flash-attn
    torch.distributed.init_process_group("gloo", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1)  # its forward logs via get_rank()
else:
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.transforms import build_eagle_processor
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
    repo = "nvidia/GR00T-N1.5-3B" if STAGE == "vla" else FT_REPOS[STAGE[3:]]
    processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    vlm = GR00T_N1_5.from_pretrained(repo, torch_dtype=torch.bfloat16).backbone.eagle_model.to("cuda", torch.bfloat16).eval()
lm = vlm.language_model
lm.config._attn_implementation = "eager"                      # eager attention returns the weights we hook
image_token = vlm.config.image_token_index

captured = []
for layer in lm.model.layers:
    layer.self_attn.register_forward_hook(lambda m, i, o: captured.append(o[1]))

def encode(question, img):
    """Same chat format GR00T builds for its backbone: [image] + text, generation prompt appended."""
    conv = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    images, _ = processor.process_vision_info(conv)
    enc = processor(text=[text], images=images, return_tensors="pt", padding=True)
    if STAGE == "vlm":                                        # Eagle2-2B's forward needs a per-tile validity flag
        enc["image_flags"] = torch.ones(enc["pixel_values"].shape[0], 1, dtype=torch.long)
    return {k: (v.to("cuda", torch.bfloat16) if v.is_floating_point() else v.to("cuda")) for k, v in enc.items()}

rows = []
for scene in load_scenes():
    for obj in scene["objects"]:
        question = f"where is the {obj['name']}"             # GR00T lowercases and strips punctuation
        enc = encode(question, scene["img"])
        ids = enc["input_ids"][0].tolist()
        img_pos = [i for i, t in enumerate(ids) if t == image_token]
        side = int(round(len(img_pos) ** 0.5))
        assert side * side == len(img_pos), f"image tokens {len(img_pos)} are not a square grid"
        name_ids = processor.tokenizer(" " + obj["name"], add_special_tokens=False)["input_ids"]
        q = next(list(range(s, s + len(name_ids))) for s in range(len(ids)) if ids[s:s + len(name_ids)] == name_ids)
        captured.clear()
        with torch.no_grad():
            vlm(**{k: v for k, v in enc.items() if k != "image_sizes"}, output_attentions=True)
        att = torch.stack([a[0][:, q][:, :, img_pos].float().mean(dim=(0, 1)) for a in captured])   # (layers, patches)
        grid = att.mean(0).reshape(side, side).cpu().numpy()
        hit, lift, peak = score(grid, obj["mask_arr"])
        rows.append(dict(scene=scene["suite"], key=scene["key"], label=obj["name"], img=scene["img"],
                         mask=obj["mask_arr"], grid=grid, hit=hit, lift=lift, peak=peak))
        print(scene["suite"], obj["name"], f"| layers {len(captured)} grid {side}x{side} | hit {hit} lift {lift:.2f}")

save_grids(rows, "groot", STAGE, OUT)          # per-scene comparison figures are drawn by compare_heatmaps.py
save_summary(rows, "groot", STAGE, OUT)
