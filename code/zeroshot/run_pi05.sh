#!/bin/bash
# pi0.5, no LIBERO fine-tuning: VLA heatmaps + arm test with pi05_base, then delete it and run the PaliGemma VLM heatmaps.
cd /workspace/libero_run
export HF_HOME=/workspace/.hf_home PYTHONPATH=$PWD/zeroshot
echo "=== pi05 vla heatmaps ==="
env_pi/bin/python zeroshot/heatmap_paligemma.py vla 2>&1 | grep -v -i warn
echo "=== pi05 arm test ==="
PORT=5555 env_pi/bin/python zeroshot/policy_pi05.py > pi05_server.log 2>&1 &
SERVER=$!
until grep -q "policy server ready" pi05_server.log || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
PYTHONPATH=$PWD/LIBERO:$PWD/zeroshot MUJOCO_GL=egl env/bin/python zeroshot/libero_client.py --port 5555 --name pi05_base 2>&1 \
  | grep -v -i -E "warn|gym|migration|upgrade"
kill $SERVER; tail -20 pi05_server.log | grep -i -E "error|traceback" -A3
rm -rf $HF_HOME/hub/models--lerobot--pi05_base; find $HF_HOME/hub -size +100M -delete
echo "=== paligemma vlm heatmaps ==="
env_pi/bin/python zeroshot/heatmap_paligemma.py vlm 2>&1 | grep -v -i warn
rm -rf $HF_HOME/hub/models--google--paligemma-3b-pt-224/blobs/*; find $HF_HOME/hub -size +100M -delete
env/bin/python zeroshot/compare_heatmaps.py pi05
echo ALL_DONE
