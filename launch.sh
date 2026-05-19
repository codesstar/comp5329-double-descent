#!/bin/bash
# 一键启动全部实验，后台运行
# 用法：bash launch.sh

cd "$(dirname "$0")"

SESSION="dl-exp"
VENV=".venv/bin/python"
LOG="results/train.log"

mkdir -p results

# 如果 session 已存在则先关掉
tmux kill-session -t $SESSION 2>/dev/null

echo "启动实验..."
echo "tmux session: $SESSION"
echo "日志文件: $LOG"
echo ""
echo "查看进度: python monitor.py"
echo "查看日志: tmux attach -t $SESSION"
echo "退出查看: Ctrl+B 然后按 D（不会停止训练）"
echo ""

# 启动 tmux 后台 session
tmux new-session -d -s $SESSION \
  "PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 caffeinate -i $VENV -m experiments.run_all 2>&1 | tee $LOG; \
   osascript -e 'display notification \"所有实验完成！\" with title \"Deep Learning\"'"

echo "实验已在后台启动，关掉终端也不影响运行。"
