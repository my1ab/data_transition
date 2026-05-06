# 均有自动创建环境
export TMPDIR=/root/autodl-tmp/tmp
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

# pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
# pip3 install flash-attn==2.7.4.post1 --no-build-isolation

cd /home/dpepo/Code-for-DPEPO-main
pip3 install -e .
cd /home/dpepo/Code-for-DPEPO-main/my_scripts/install

pip3 install vllm==0.8.5

pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5


pip install numpy==2.2.0 --force-reinstall

alfworld-download -f
# alfworld-play-tw