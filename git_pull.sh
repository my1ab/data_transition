#!/bin/bash

# Git 拉取仓库文件脚本

set -e

echo "=== 设置 Git 用户信息 ==="
git config user.name "my1ab"
git config user.email "my1ab@example.com"

echo ""
echo "=== 检查当前目录 ==="
pwd

echo ""
echo "=== 添加远程仓库 ==="
git remote add my-verl https://github.com/my1ab/data_transition.git || echo "远程仓库已存在"

echo ""
echo "=== 拉取仓库文件 ==="
git pull my-verl main

echo ""
echo "=== 拉取完成 ==="