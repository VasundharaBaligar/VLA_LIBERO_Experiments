"""Stage 1 check: OpenVLA-OFT module structure and the exact token positions the action head reads, without downloading
the weights (config + tokenizer only; the model is built on the meta device).
Usage: python inspect_model.py"""
import os, sys
from unittest import mock

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from transformers import AutoImageProcessor

# Inspection only: `import prismatic` pulls in the RLDS training loader (TensorFlow, dlimp, tfds). Building the model
# never touches them, so placeholders let this run before TensorFlow is installed. Not used for collection.
for _m in ("tensorflow", "tensorflow_datasets", "tensorflow_graphics", "tensorflow_graphics.geometry",
           "tensorflow_graphics.geometry.transformation", "dlimp"):
    sys.modules.setdefault(_m, mock.MagicMock())

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

CKPT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    path = snapshot_download(CKPT, allow_patterns=["*.json", "*.py", "tokenizer*", "*.model"])
    print("downloaded (no weights):", sorted(os.listdir(path)))
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with init_empty_weights():
        vla = OpenVLAForActionPrediction(cfg)
    print("\n== top-level modules")
    for n, m in vla.named_children():
        print(f"  {n}: {type(m).__name__}  ({sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params)")
    lm = vla.language_model
    layers = lm.model.layers
    print("\n== language model:", type(lm).__name__, "| transformer layers at vla.language_model.model.layers:",
          len(layers), "| hidden size:", lm.config.hidden_size)
    print("  one layer:", layers[0])
    print("\n== vision backbone:", type(vla.vision_backbone).__name__, "| featurizers:",
          [n for n, _ in vla.vision_backbone.named_children()])
    print("  patches per image:", vla.vision_backbone.get_num_patches(), "| images in input (eval default): 2")
    tok = AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer
    prompt = f"In: What action should the robot take to {TASK.lower()}?\nOut:"
    ids = tok(prompt)["input_ids"] + [29871]                  # predict_action appends the empty token 29871
    P = len(ids) - 1                                          # NUM_PROMPT_TOKENS (excludes <BOS>)
    N = 2 * vla.vision_backbone.get_num_patches() + 1          # 2 images + 1 proprio token
    A = ACTION_DIM * NUM_ACTIONS_CHUNK
    print(f"\n== sequence for task 0: <BOS>=0 | patches+proprio={1}..{N} | prompt={N + 1}..{N + P} | "
          f"action placeholders={N + P + 1}..{N + P + A} | stop={N + P + A + 1} | length={N + P + A + 2}")
    print(f"  action head reads positions [{N + P}, {N + P + A}) = {A} positions "
          f"(NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK} x ACTION_DIM={ACTION_DIM})")
    print(f"  last token = position {N + P + A + 1} (the </s> stop token)")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
