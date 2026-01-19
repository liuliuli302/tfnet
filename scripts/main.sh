# !/bin/bash
# 导出PYTHONPATH为workspace
export PYTHONPATH=$(pwd):$PYTHONPATH

python src/main.py
