# 开始时间


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Base Import
import os 
import sys
import json  
import ray
import time
import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent'))
# Add webshop directory to sys.path for web_agent_site module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verl-agent/agent_system/environments/env_package/webshop/webshop'))

# WebShop Import
import gym
from agent_system.environments.env_package.webshop import webshop_projection
# /home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py

# from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS as AGENT_WEBSHOP_TEMPLATE_NO_HIS
# from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE as AGENT_WEBSHOP_TEMPLATE
# 引入本地官方 prompt
from webshop import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE, system_message
from prompts_webshop import system_message_para, reason_prompt_para
from prompts_webshop2 import system_message_para2, reason_prompt_para2
# /home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py
from agent_system.environments.env_package.webshop.webshop.web_agent_site.envs.web_agent_text_env \
import WebAgentTextEnv
from agent_system.environments.env_package.webshop import build_webshop_envs
print("import WebAgentTextEnv success")

# exit(0)

remote = 0

def deepseek(messages):
    # sk-05267e6863d6455eb1a8c2fc92df3005
    client = OpenAI(api_key="sk-05267e6863d6455eb1a8c2fc92df3005", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )
    
    return response.choices[0].message.content 

def test_api_connection():
    """
    Test function to verify the remote API connection works correctly.
    Sends a simple test message and prints the response.
    """
    print("Testing DeepSeek API connection...")
    
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Hello! Please respond with a simple greeting."}
            {"role": "user", "content": "We are testing whether the API connection works."}
        ]
        
        response = deepseek(test_messages)
        print(f"API Response: {response}")
        print("API connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"API connection test FAILED: {e}")
        print("Please check your API key and network connection.")
        return False


# Util Functions
import re
from openai import OpenAI


def extract_think_and_action(text, use_para=0, num_para=1, total_envs=1):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    actions_dict = {idx: "null" for idx in range(total_envs)}
    
    env_pattern = r'<env_(\d+)>(.*?)</env_\d+>'
    matches = re.findall(env_pattern, text, re.DOTALL)
    
    for env_index, action in matches:
        env_index = int(env_index)
        if 0 <= env_index <= total_envs - 1:
            action = action.strip()
            if action and action != 'None' and action != 'null':
                actions_dict[env_index] = action
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

# 根据任务描述+动作字符串规范动作
def read_json(file_path):
    data = json.load(open(file_path,'r'))
    return data

def get_unique_filename(file_path):
    """
    Check if a file exists and generate a unique filename by appending a number if needed.
    
    Args:
        file_path: Original file path
        
    Returns:
        Unique file path
    """
    if not os.path.exists(file_path):
        return file_path
    
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    
    counter = 1
    while True:
        new_file_name = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name) if dir_name else new_file_name
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1

# Main Logic - Generate coldstart data for WebShop
def get_single_trajectory(env, task_idx, env_idx=0, turns=50, show_turn=False, use_history=True,
                          use_para=0, num_para=1, env_num=50, group_n=1, prompt=1):
    """
    Generate a single trajectory for WebShop environment with multiple turns
    Returns messages in the format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    
    Args:
        show_turn: If True, adds turn number markers to message contents
        use_history: If True, use WEBSHOP_TEMPLATE with history, else use WEBSHOP_TEMPLATE_NO_HIS
        use_para: If > 0, use parallel environment mode with <env_i> tags
        num_para: Number of parallel environments to select each turn
        env_num: Number of environments per group
        group_n: Number of groups
    """
    messages = []
    success_flag = 0
    
    # Reset the environment
    obs_list, info_list = env.reset()
    
    # Initialize observations and infos for all parallel environments
    total_envs = group_n * env_num
    obs_parallel = obs_list[:total_envs]
    info_parallel = info_list[:total_envs]
    task_descriptions = [info.get('goal', 'Find and purchase a product') for info in info_parallel]
    available_actions_parallel = [info.get('available_actions', []) for info in info_parallel]
    
    # 历史记录列表
    action_history = []
    history_length = 5  # 保留最近5步历史
    
    # Add system message - only once at the beginning
    if prompt == 2:
        messages.append({"role": "system", "content": system_message_para2.format(num_parallel=num_para, total_envs=total_envs)})
    else:
        messages.append({"role": "system", "content": system_message_para.format(num_parallel=num_para, total_envs=total_envs)})
    
    for turn in range(turns):
        # Build user prompt with task description, current observation, and available actions
        # 多环境并行模式：构建包含所有环境观察的prompt
        obs_prompt = ''
        for idx in range(total_envs):
            admissible_commands = "\n".join([f"  - {action}" for action in available_actions_parallel[idx]])
            obs_prompt += f'<observation_{idx}>\nThe observation and next candidated actions of {idx}-th environment are:\nObservation:\n{obs_parallel[idx]}\nNext Possible Actions:\n{admissible_commands}\n</observation_{idx}>\n'
        
        if prompt == 2:
            message_sent = reason_prompt_para2.format(
                task_description=task_descriptions[0],
                current_observation=obs_prompt,
                admissible_actions="\n".join([f"  - {action}" for actions in available_actions_parallel for action in actions]),
                num_parallel=num_para,
                total_envs=total_envs
            )
        else:
            message_sent = reason_prompt_para.format(
                task_description=task_descriptions[0],
                current_observation=obs_prompt,
                admissible_actions="\n".join([f"  - {action}" for actions in available_actions_parallel for action in actions]),
                num_parallel=num_para,
                total_envs=total_envs
            )
        
        # Add user message (user提出需求)
        messages.append({"role": "user", "content": message_sent})
        
        output = deepseek(messages=messages)
        result = extract_think_and_action(output, use_para=use_para, num_para=num_para, total_envs=total_envs)
        
        # Build assistant_response based on mode
        # 多环境并行模式
        if result['think'] and result.get('actions'):
            actions_str = "\n".join([f"<env_{k}>{v}</env_{k}>" for k, v in result['actions'].items()])
            assistant_response = f"<think>\n{result['think']}\n</think>\n\n<parallel>\n{actions_str}\n</parallel>"
        else:
            assistant_response = f"<think>\nNo reasoning available\n</think>\n\n<parallel>\n</parallel>"
        
        # Execute action and get next observation
        normalized_actions = None
        done_list = [False] * total_envs
        reward_list = [None] * total_envs
        action_valid_list = [False] * total_envs
        
        # 多环境并行模式动作执行
        if result.get('actions') and len(result['actions']) > 0:
            normalized_actions = []
            for idx in range(total_envs):
                # 0开
                action = result['actions'].get(idx,'empty action')
                normalized_actions.append(action)
            
            # 过滤有效动作并执行
            # valid_actions = normalized_actions.copy()
            # 兼容不同环境的 step 返回值
            step_result = env.step(normalized_actions)
            if len(step_result) == 6:
                obs_list, reward_list, done_list, info_list, rewards, action_valids = step_result
            elif len(step_result) == 5:
                obs_list, reward_list, done_list, info_list, action_valids = step_result
                # rewards = reward_list[0] if reward_list else None
            else:
                obs_list, reward_list, done_list, info_list = step_result
                # rewards = reward_list[0] if reward_list else None
                action_valids = True
            
            # 更新 action_valid_list
            if isinstance(action_valids, (list, tuple)):
                action_valid_list = list(action_valids[:total_envs])
            else:
                action_valid_list = [action_valids] * total_envs
            # 更新 reward_list（截断到 total_envs）
            reward_list = reward_list[:total_envs]
            
            # 更新每个环境的状态
            obs_parallel = obs_list[:total_envs]
            info_parallel = info_list[:total_envs]
            task_descriptions = [info.get('goal', 'Find and purchase a product') for info in info_parallel]
            available_actions_parallel = [info.get('available_actions', []) for info in info_parallel]
            
            # 记录历史
            # for idx in range(total_envs):
            #     if normalized_actions[idx]:
            #         action_history.append({
            #             'obs': obs_parallel[idx],
            #             'action': normalized_actions[idx]
            #         })
        else:
            # 没有有效动作，reward_list为 None
            # rewards = [None] * total_envs
            reward_list = [None] * total_envs
        
        # Add assistant message (assistant作回复)
        # for idx in range(total_envs):
                # # 0开
                # action = result['actions'].get(idx)
                # normalized_actions.append(action)
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "rewards": json.dumps(reward_list),
            "original_actions": result.get('actions', {}),
            # "normalized_actions": normalized_actions,
        }
        
        if show_turn:
            assistant_msg["turn"] = turn + 1
        messages.append(assistant_msg)
        
        # Check if task is done
        # 多环境模式：任一环境完成或所有环境都无效则结束
        any_done = any(done_list[:total_envs])
        all_invalid = not normalized_actions or all(a is None or a == 'null' or a == 'None' for a in normalized_actions)

        if any_done:
            completed_idx = [i for i, done in enumerate(done_list[:total_envs]) if done]
            # 检查 reward_list 中只要有一个大于0即设定 success_flag
            if reward_list:
                for reward in reward_list:
                    if reward is not None and reward > 0:
                        success_flag = 1
                        break
            status = "SUCCESS" if success_flag == 1 else "FAILED"
            status_msg = f"Task {task_idx} {status} at turn {turn + 1} in environments {completed_idx}"
            print(status_msg)
            break
        elif all_invalid:
            status_msg = f"Task {task_idx} exit(all null) at turn {turn + 1}"
            print(status_msg)
            break
    
    return messages, success_flag, status_msg

def generate_coldstart_data(output_file, num_cpus=0.1, num_samples=500, turns=50, use_ray=False, use_para=0, show_turn=False, 
                            use_history=True,load_all = 0, human_goals=False, num_para=1, group_n=1, env_num=50,
                            only_test=0, prompt=1):
    """
    Generate coldstart data for WebShop in JSON format.
    
    Args:
        output_file: Path to save the generated data
        num_samples: Number of samples to generate
        turns: Maximum number of turns per trajectory
        show_turn: If True, adds turn number markers to message contents (default: False)
        use_history: If True, use WEBSHOP_TEMPLATE with history, else use WEBSHOP_TEMPLATE_NO_HIS (default: True)
    """
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    print(f'Start time: {start_time_str}')
    print(f'Generating {num_samples} samples with {turns} turns each...')
    
    expected_num_para = group_n * env_num
    # print(f"num_para ({num_para}) > group_n * env_num ({group_n} * {env_num} = {expected_num_para})")
    if num_para > expected_num_para:
        print(f"num_para ({num_para}) > group_n * env_num ({group_n} * {env_num} = {expected_num_para})")
        print("exiting...")
        exit(1)
    
    env_start_time = time.time()
    # Get unique filename if file exists
    output_file = get_unique_filename(output_file)
    log_file = output_file.replace('.json', '_success.txt')
    
    # base_path = '/home/dpepo/verl-agent/agent_system/environments/env_package/webshop/webshop/data/'
    if remote == 0:
        base_path = '/home/dpepo/data/'
    else:
        base_path = '/diskpool/home/xuxz/data/'    

    if load_all == 0:
        file_path = os.path.join(base_path, 'items_shuffle_1000.json')
        attr_path = os.path.join(base_path, 'items_ins_v2_1000.json')
    else:
        file_path = os.path.join(base_path, 'items_shuffle.json')
        attr_path = os.path.join(base_path, 'items_ins_v2.json')

    print(f'file_path: {file_path} \n attr_path: {attr_path}')
    
    env_kwargs = {
        'observation_mode': 'text',
        'num_products': None,
        'human_goals': human_goals,
        'file_path': file_path,
        'attr_path': attr_path
    }
    
    # Build environment - parallel mode
    # SFT数据生成使用 'sft' split（1500-4000），与RL训练（4000-）完全隔离
    env = build_webshop_envs(
        seed=42,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker={'num_cpus': num_cpus},
        is_train=True,
        split='sft',
        env_kwargs=env_kwargs
    )
    
    print(f'Environment built, took {time.time() - env_start_time} seconds')
    
    # Generate data
    coldstart_data = []
    success_data = []  # 存储成功的轨迹（格式与原message相同）
    
    if only_test == 1:
        env.close()
        print('only_test True')
        return

    for i in tqdm(range(num_samples), desc="Generating coldstart data"):
        try:
            # 得到单条轨迹
            # env = WebshopSingleEnv
            trajectory, success_flag, status_msg = get_single_trajectory(env, task_idx=i, env_idx=0, turns=turns, use_para=use_para, 
                                               show_turn=show_turn, num_para=num_para, use_history=use_history,
                                               env_num=env_num, group_n=group_n, prompt=prompt)
            # For non-Ray mode, each trajectory is a messages list
            coldstart_data.append(trajectory)
            
            # 将状态信息写入日志文件
            with open(log_file, 'a') as f:
                f.write(status_msg + '\n')
            
            # 如果成功，添加到成功轨迹列表
            if success_flag == 1:
                success_data.append(trajectory)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coldstart_data, f, indent=4)
    
    print(f"Coldstart data saved to {output_file}")
    print(f"Total {len(coldstart_data)} entries generated")
    
    # 保存成功轨迹到新的 JSON 文件（格式与原message相同）
    if success_data:
        success_output_file = output_file.replace('.json', '_success.json')
        with open(success_output_file, 'w') as f:
            json.dump(success_data, f, indent=4)
        
        success_count = len(success_data)
        success_rate = success_count / len(coldstart_data) if coldstart_data else 0
        
        print(f"Success trajectories saved to {success_output_file}")
        print(f"Success count: {success_count}, Success rate: {success_rate:.2%}")
        
        # 追加写入日志文件（不覆盖之前的状态信息）
        # w则覆盖 a则添加
        with open(log_file, 'a') as f:
            f.write("\n")
            f.write(f"Success trajectories saved to {success_output_file}\n")
            f.write(f"Success count: {success_count}, Success rate: {success_rate:.2%}\n")
            f.write(f"Total entries: {len(coldstart_data)}\n")
            f.write(f"Mode: parallel\n")
            f.write(f"Number of parallel environments: {num_para}\n")
    else:
        print("No success trajectories generated.")
        # 追加写入日志文件（不覆盖之前的状态信息）
        with open(log_file, 'a') as f:
            f.write("\n")
            f.write("No success trajectories generated.\n")
        
    # Cleanup
    env.close()
    
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f'Start time: {start_time_str}')
    print(f'End time: {end_time_str}')
    print(f'Elapsed time: {elapsed_time}s')
    
    # 将用时信息写入日志文件
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write(f"Start time: {start_time_str}\n")
            f.write(f"End time: {end_time_str}\n")
            f.write(f"Elapsed time: {elapsed_time}s\n")
    
    return coldstart_data

if __name__ == "__main__":
    # Configuration
    # /home/dpepo/verl-agent/coldstart_genaration_webshop/webshop_coldstart.py
    if remote == 0:
        OUTPUT_FILE = f'/home/dpepo/verl-agent/coldstart_result_webshop/WebShop_coldstart.json'  
    else:
        OUTPUT_FILE = f'/diskpool/home/xuxz/verl-agent/coldstart_result_webshop/WebShop_coldstart.json'
        
    if OUTPUT_FILE and not os.path.exists(os.path.dirname(OUTPUT_FILE)):
            os.makedirs(os.path.dirname(OUTPUT_FILE))
    
    # Generate coldstart data
    move = 2
    if move == 1:
        test_api_connection()
    elif move == 2:
        test_api_connection()
        
        generate_coldstart_data(OUTPUT_FILE, num_samples=30, turns=20, 
                                # use_ray=False, 
                                num_cpus=0.5,
                                use_para=1, num_para=4, group_n=8, env_num=1,
                                show_turn=1, use_history=0, load_all=0, 
                                # human_goals=True, 
                                human_goals=False,
                                # only_test=1,
                                prompt=1
                                )
        # print(f'====== gap between two sets =======')
        # generate_coldstart_data(OUTPUT_FILE, num_samples=30, turns=20, 
        #                         # use_ray=False, 
        #                         num_cpus=0.5,
        #                         use_para=1, num_para=4, group_n=1, env_num=8,
        #                         show_turn=1, use_history=0, load_all=0, 
        #                         # human_goals=True, 
        #                         human_goals=False,
        #                         # only_test=1,
        #                         prompt=2
        #                         )
        
        # show_turn为bool型 但可以赋值为1/0 true表示所有非0数