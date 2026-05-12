import bisect
import hashlib
import logging
import random
import os
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

# /home/dpepo/Code-for-DPEPO-main/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py
# 默认数据集位置 冷启动时可修改
# 执行时由verl-agent/examples/data_preprocess/prepare.py生成parquet
DEFAULT_ATTR_PATH = '/home/dpepo/data/items_ins_v2_1000.json'
DEFAULT_FILE_PATH = '/home/dpepo/data/items_shuffle_1000.json'
DEFAULT_REVIEW_PATH = '/home/dpepo/data/reviews.json'

FEAT_CONV = '/home/dpepo/data/feat_conv.pt'
FEAT_IDS = '/home/dpepo/data/feat_ids.pt'

HUMAN_ATTR_PATH = '/home/dpepo/data/items_human_ins.json'
# HUMAN_ATTR_PATH = os.environ.get('HUMAN_ATTR_PATH', '/home/dpepo/data/items_human_ins.json')
# HUMAN_ATTR_PATH = os.environ.get('HUMAN_ATTR_PATH', '/home/dpepo/data/splited/items_human_ins_cleaned_split.json')

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()