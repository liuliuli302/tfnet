#!/usr/bin/env bash
# ============================================================
# 一键运行消融实验，输出与毕业论文 c3.tex 中
#   tab:ablation_all   (模块消融)
#   tab:alpha_ablation  (参数 α 扫描)
# 完全一致的结果表格。
#
# 用法:
#   bash scripts/ablation.sh            # 默认: 带时序平滑 (论文主表)
#   bash scripts/ablation.sh --no-smoothing  # 不带平滑的对照
#   bash scripts/ablation.sh --mode module   # 仅运行模块消融
#   bash scripts/ablation.sh --mode alpha    # 仅运行 α 扫描
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

python ablation/run_module_ablation.py "$@"
