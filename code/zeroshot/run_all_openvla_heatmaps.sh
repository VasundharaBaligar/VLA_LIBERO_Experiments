#!/bin/bash
# All OpenVLA stages on every scene object, one model on disk at a time (download -> test -> delete).
# STAGES env var picks a subset, e.g. STAGES="ft_libero_10".
cd /workspace/libero_run
export HF_HOME=/workspace/.hf_home
for st in ${STAGES:-vlm vla ft_libero_spatial ft_libero_object ft_libero_goal ft_libero_10}; do
  echo "=== $st ==="
  env/bin/python zeroshot/heatmap_openvla.py $st 2>&1 | grep -E "hits|Error|error|Traceback|^ +File" | grep -v "^    {"
  rm -rf $HF_HOME/hub/models--openvla--openvla-7b-finetuned-* $HF_HOME/hub/models--TRI-ML--prismatic-vlms $HF_HOME/hub/models--timm--*
  find $HF_HOME/hub -size +100M -delete      # weights sit in hub/blobs or hub/models--*/blobs
done
env/bin/python zeroshot/compare_heatmaps.py openvla
echo ALL_DONE
