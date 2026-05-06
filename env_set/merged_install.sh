#!/bin/bash
# Merged install script from alfworld, sciworld, and webshop
# Created by comparing install_alfworld.sh, install_sciworld.sh, and install_webshop.sh

# ============================================
# Environment Setup (ignored per request)
# ============================================
# Original conda environment creation has been removed as requested

# ============================================
# Common Packages (all environments)
# ============================================
# conda create -n dpepo_webshop python=3.11.15
# conda activate dpepo_webshop

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip3 install torchvision==0.15.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Core packages
# vllm==0.8.5: matches requirement.txt ✓
pip3 install vllm==0.8.5

# gymnasium==0.29.1: NOT in requirement.txt
pip3 install gymnasium==0.29.1

# stable-baselines3==2.6.0: matches requirement.txt ✓
pip3 install stable-baselines3==2.6.0

pip install gdown==6.0.0
# alfworld: NOT in requirement.txt



# Alfworld Specific
# numpy==2.2.0: CONFLICT with requirement.txt (numpy==1.26.4)
# pip install numpy==2.2.0 --force-reinstall
pip install numpy==1.26.4 
# 最后再装
pip3 install flash-attn==2.7.4.post1  --no-build-isolation --no-cache-dir
pip install alfworld
alfworld-download -f

# Sciworld Specific
# scienceworld: requirement.txt specifies scienceworld==1.2.3

# ray==2.49.1: CONFLICT with requirement.txt (ray==2.47.1)
pip install ray==2.47.1
pip install scienceworld

# 后期检查修复
pip install omegaconf
pip install python-Levenshtein

# Webshop Specific
cd ./agent_system/environments/env_package/webshop/webshop
chmod +x setup.sh
./setup.sh -d all

# Download spaCy large NLP model
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_sm
# 取代以上安装 消除404问题
conda install -c conda-forge spacy-model-en_core_web_lg
conda install -c conda-forge spacy-model-en_core_web_sm


# ============================================
# VERSION CONFLICTS DETECTED
# ============================================
# 
# WARNING: vllm version conflict detected:
# - alfworld.sh requires: vllm==0.8.5
# - sciworld.sh requires: vllm==0.8.5  
# - webshop.sh requires: vllm==0.8.2
# 
# Current choice: Using vllm==0.8.5 (majority)

# ============================================
# VERSION CONFLICTS WITH env_set/requirement.txt
# ============================================
# 
# Package conflicts:
# 1. numpy: install scripts require ==2.2.0, requirement.txt has ==1.26.4
# 2. ray: install scripts require ==2.49.1, requirement.txt has ==2.47.1
# 
# Packages in install scripts NOT in requirement.txt:
# 1. gymnasium==0.29.1
# 2. alfworld (no version specified)
# 
# Package with version mismatch:
# - scienceworld: install scripts don't specify version, requirement.txt has ==1.2.3

# spacy 3.7.2 requires typer<0.10.0,>=0.3.0
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0
# 原环境typer==0.9.4以解决冲突