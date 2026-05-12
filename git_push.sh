#!/bin/bash

# git remote add my-verl https://github.com/my1ab/data_transition.git
# git remote -v

# Git 日常提交和推送到远端仓库脚本

set -e

echo "=== 设置 Git 用户信息 ==="
git config user.name "my1ab"
git config user.email "my1ab@example.com"

echo ""
echo "=== 检查当前目录 ==="
pwd

echo ""
echo "=== 检查 Git 状态 ==="
git status

echo ""
echo "=== 检查并创建目标分支 ==="
# 手动选择目标分支
TARGET_BRANCH="main"
# TARGET_BRANCH="my-verl"

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    echo "分支 $TARGET_BRANCH 已存在"
else
    echo "分支 $TARGET_BRANCH 不存在，创建该分支"
    git branch $TARGET_BRANCH
fi

echo ""
echo "=== 添加所有已追踪文件 ==="
git add -u

echo ""
echo "=== 添加所有新文件 ==="
git add .

echo ""
echo "=== 排除不需要的文件夹和大文件 ==="
EXCLUDE_PATHS=(
    "model/"
    "coldstart_example/"
    "data/"
    "coldstart_result_webshop/"
    # "coldstart_genaration_webshop/"
    "*.pt"
    "*.ckpt"
    "*.safetensors"
    "*.tar.gz"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
)

for path in "${EXCLUDE_PATHS[@]}"; do
    echo "排除: $path"
    git reset HEAD "$path" 2>/dev/null || true
done

echo ""
echo "=== 检查暂存状态 ==="
git status


# echo ""
# echo "=== 暂存区大小统计 ==="
# git diff --cached --stat

# echo ""
# echo "=== 暂存区总大小 ==="
# TOTAL_SIZE=$(git diff --cached --numstat | awk '{sum+=$1+$2} END {print sum/1024/1024}')
# echo "总大小: $TOTAL_SIZE MB"

echo ""
echo "=== 提交更改 ==="
git commit -m "Update project files"

echo ""
echo "=== 推送到远端仓库 $TARGET_BRANCH 分支 ==="
# 格式: git push <远程名> <来源>:<目标> -f
git push my-verl HEAD:$TARGET_BRANCH -f

echo ""
echo "=== 操作完成 ==="