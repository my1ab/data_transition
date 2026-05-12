s#!/bin/bash

# 设置 HUMAN_ATTR_PATH 环境变量（可根据需要修改）
export HUMAN_ATTR_PATH="/home/dpepo/data/items_human_ins.json"

# 切换到 coldstart_genaration_webshop 目录
cd /home/dpepo/verl-agent/coldstart_genaration_webshop

# 运行 coldstart_para.py
python coldstart_para.py