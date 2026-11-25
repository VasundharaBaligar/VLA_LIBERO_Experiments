#!/bin/bash
# GR00T N1.5, no LIBERO fine-tuning: arm test with the base model, then heatmaps for the LIBERO fine-tuned checkpoints
# and the Eagle2-2B VLM stand-in. One model on disk at a time (download -> test -> delete).
cd /workspace/libero_run
export HF_HOME=/workspace/.hf_home PYTHONPATH=$PWD/zeroshot:$PWD/shims
echo "=== groot arm test ==="
PORT=5556 env_groot/bin/python zeroshot/policy_groot.py > groot_server.log 2>&1 &
SERVER=$!
until grep -q "policy server ready" groot_server.log || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
PYTHONPATH=$PWD/LIBERO:$PWD/zeroshot MUJOCO_GL=egl env/bin/python zeroshot/libero_client.py --port 5556 --name groot_base 2>&1 \
  | grep -v -i -E "warn|gym|migration|upgrade"
kill $SERVER; grep -i -E "error|traceback" -A3 groot_server.log | tail -20
rm -rf $HF_HOME/hub/models--nvidia--GR00T-N1.5-3B; find $HF_HOME/hub -size +100M -delete
for st in ft_libero_spatial ft_libero_object ft_libero_goal ft_libero_10 vlm; do
  echo "=== groot $st heatmaps ==="
  env_groot/bin/python zeroshot/heatmap_groot.py $st 2>&1 | grep -E "hits|Error|error|Traceback|^ +File" | grep -v "^    {"
  rm -rf $HF_HOME/hub/models--youliangtan--* $HF_HOME/hub/models--nvidia--Eagle2-2B; find $HF_HOME/hub -size +100M -delete
done
env/bin/python zeroshot/compare_heatmaps.py groot
echo ALL_DONE
