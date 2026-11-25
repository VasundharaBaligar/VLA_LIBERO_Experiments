#!/bin/bash
# Heatmaps for each LIBERO fine-tuned OpenVLA, one at a time (download -> test -> delete) to fit the 32 GB disk.
cd /workspace/libero_run
export HF_HOME=/workspace/.hf_home
for s in libero_spatial libero_object libero_goal libero_10; do
  echo "=== $s ==="
  env/bin/python zeroshot/heatmap_openvla.py ft_$s 2>&1 | grep -v Warning
  rm -rf $HF_HOME/hub/models--openvla--openvla-7b-finetuned-${s//_/-}
  find $HF_HOME/hub/blobs -size +500M -delete
done
echo ALL_DONE
