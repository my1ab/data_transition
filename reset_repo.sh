#!/bin/bash

# Git 完全重置仓库脚本
# 清除所有历史，创建全新提交

set -e

echo "=== 设置 Git 用户信息 ==="
git config user.name "my1ab"
git config user.email "my1ab@example.com"

echo ""
echo "=== 检查当前目录 ==="
pwd

echo ""
echo "=== 警告：此操作将完全清除所有 Git 历史！ ==="
read -p "确定要继续吗？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 1
fi

echo ""
echo "=== 完全清除 Git 历史 ==="
echo "正在移除 .git 目录..."
rm -rf .git

echo ""
echo "=== 重新初始化 Git 仓库 ==="
git init -b main

echo ""
echo "=== 添加远程仓库 ==="
git remote add my-verl https://github.com/my1ab/data_transition.git

echo ""
echo "=== 清空远端仓库所有文件 ==="
echo "正在创建空提交并强制推送以清空远端仓库..."

# echo ""
# echo "=== 检查大文件 ==="
# echo "查找超过 100MB 的文件..."
# large_files=$(find . -type f -size +99M 2>/dev/null)
# if [ -n "$large_files" ]; then
#     echo "发现大文件:"
#     echo "$large_files"
#     echo ""
#     echo "=== 移除大文件 ==="
#     echo "正在删除超过 100MB 的文件以符合 GitHub 限制..."
#     find . -type f -size +99M -delete
# else
#     echo "未发现超过 100MB 的文件"
# fi

# echo ""
# echo "=== 移除远端仓库所有文件 ==="
# echo "正在强制推送空提交以清空远端仓库..."
# git add -A
# git commit -m "Initial commit"

git commit --allow-empty -m "Empty commit to clear remote"
git push -f my-verl main