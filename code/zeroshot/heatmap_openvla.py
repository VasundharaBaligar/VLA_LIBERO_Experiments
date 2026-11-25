# Object-location attention heatmaps for OpenVLA's VLA (openvla-7b) and its VLM (Prismatic dinosiglip-224px+7b).
# Both share one architecture, so the VLM is built from the OpenVLA config and loaded with the Prismatic weights.
# Usage: python heatmap_openvla.py vla | vlm
import os, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
from heatmap_common import load_scenes, score, save_grids, save_summary

STAGE = sys.argv[1]                       # vla | vlm | ft_<suite> (e.g. ft_libero_10 -> openvla-7b-finetuned-libero-10)
OUT = "/workspace/fa26-vla-cs670/zeroshot_results/heatmaps"
REPO = "openvla/openvla-7b"
VLA_REPO = f"openvla/openvla-7b-finetuned-{STAGE[3:].replace('_', '-')}" if STAGE.startswith("ft_") else REPO
processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)

def load_vlm():
    import timm
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForVision2Seq.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                   attn_implementation="eager")
    ckpt = torch.load(hf_hub_download("TRI-ML/prismatic-vlms", "prism-dinosiglip-224px+7b/checkpoints/latest-checkpoint.pt"),
                      map_location="cpu", weights_only=False)["model"]
    sd = {}
    for k, v in ckpt["llm_backbone"].items():                 # llm.model.layers... -> language_model.model.layers...
        sd["language_model." + k.removeprefix("llm.")] = v
    for k, v in ckpt["projector"].items():                    # projector.{0,2,4}.* -> projector.fc{1,2,3}.*
        _, idx, param = k.split(".")
        sd[f"projector.{ {'0': 'fc1', '2': 'fc2', '4': 'fc3'}[idx] }.{param}"] = v
    # Prismatic keeps the vision encoders frozen at their original pretrained weights
    for prefix, name in [("featurizer", "vit_large_patch14_reg4_dinov2.lvd142m"),
                         ("fused_featurizer", "vit_so400m_patch14_siglip_224")]:
        for k, v in timm.create_model(name, pretrained=True, num_classes=0, img_size=224).state_dict().items():
            sd[f"vision_backbone.{prefix}." + k.replace(".gamma", ".scale_factor")] = v
    missing, unexpected = model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False, assign=True)
    print("VLM load -> missing:", missing, "| unexpected:", unexpected[:5])
    return model

if STAGE != "vlm":
    model = AutoModelForVision2Seq.from_pretrained(VLA_REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                   trust_remote_code=True, attn_implementation="eager")
else:
    model = load_vlm()
model = model.to("cuda:0").eval()

rows = []
for scene in load_scenes():
    for obj in scene["objects"]:
        prefix = "In: Where is the "
        prompt = f"{prefix}{obj['name']}?\nOut:"
        inputs = processor(prompt, Image.fromarray(scene["img"])).to("cuda:0", dtype=torch.bfloat16)
        enc = processor.tokenizer(prompt, return_offsets_mapping=True)
        start, end = len(prefix), len(prefix) + len(obj["name"])
        obj_tok = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and b > a]
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        n_patch = model.vision_backbone.featurizer.patch_embed.num_patches          # 256 = 16x16
        # sequence = [BOS] + image patches + remaining text, so text token i sits at i + n_patch (i >= 1)
        q = [i + n_patch for i in obj_tok]
        att = torch.stack([a[0, :, q, 1:1 + n_patch].float().mean(dim=(0, 1)) for a in out.attentions])  # (layers, patches)
        grid = att.mean(0).reshape(16, 16).cpu().numpy()
        hit, lift, peak = score(grid, obj["mask_arr"])
        rows.append(dict(scene=scene["suite"], key=scene["key"], label=obj["name"], img=scene["img"], mask=obj["mask_arr"],
                         grid=grid, hit=hit, lift=lift, peak=peak))
        print(scene["suite"], obj["name"], "tokens:", [enc.tokens()[i] for i in obj_tok], "hit", hit, f"lift {lift:.2f}")

os.makedirs(OUT, exist_ok=True)
save_grids(rows, "openvla", STAGE, OUT)          # per-scene comparison figures are drawn by compare_openvla.py
save_summary(rows, "openvla", STAGE, OUT)
