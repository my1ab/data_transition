import os
import json 
import time 
import torch
from parallel_utils import * 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from prompts import limit_prompt, system_prompt, history_prompt, compressed_prompt_initial, compressed_prompt_process

def check_cuda(detail = 1):
    print(f"CUDA available: {torch.cuda.is_available()}")
    if detail == 1:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")
        print(torch.version.cuda)  
        print(torch.backends.cudnn.version())  
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_vllm_output(output):
    outputs = [] 
    for response in output:
        outputs.append(response.outputs[0].text)
    
    return outputs 

def eval(max_files = 10):
    parameters = '1.5b'
    save_path = './results'  # 修改为本地目录
    # path = 'Qwen/Qwen2.5-1.5B-Instruct' # 使用公开可用的Qwen2.5模型
    path = 'Qwen/Qwen2.5-0.5B-Instruct'
    # path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    # path = 'models/Qwen2.5-1.5B-Instruct' 
    num_parallel = 50  # 进一步减少并行度，从10减少到5
    # max_files = 10  # 设置要处理的游戏文件数量
    import os
    # 强制设置Alfworld的数据根目录为你的本地路径
    os.environ["ALFWORLD_DATA"] = "/home/dpepo/.cache/alfworld"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用一个GPU
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # 禁用数据集内部的绝对路径，强制使用相对路径加载
    os.environ["ALFWORLD_DISABLE_ABS_PATH"] = "1"
    
    check_cuda(detail = 1)
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, device_map='auto') 
    print("-"*50 + "token done" + "-"*50)
    
    # 延迟导入ray，只在需要时初始化
    # 强制限制Ray资源，禁止进程爆炸，自动处理权限/初始化
    # import ray
    ray.init(
        # num_cpus=1,        # 减少CPU进程数
        num_cpus=4,
        num_gpus=1,
        object_store_memory=1024 * 1024 * 1024 * 4,  # 减少内存使用
        ignore_reinit_error=True,
        include_dashboard=False,  # 关闭Ray面板，省资源
    )
    
    # 优化模型加载参数
    model = LLM(
        model=path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,  # 进一步降低GPU内存使用
        trust_remote_code=True,  # 🔥 关键！必须添加
        dtype="float16",  # 使用半精度以减少内存使用
        enforce_eager=True
        # max_model_len=1024,  # 进一步减少最大上下文长度
    )
    
    print("-"*50 + "llm import done" + "-"*50)
    
    # game_path = './data_pipelines/gamefiles/alfworld/gamefiles_eval.json'  # 修改为本地路径
    game_path = './data_pipelines/gamefiles/alfworld/gamefiles_eval.json'  # 修改为本地路径
    
    data = read_json(game_path) 
    
    todo_files = [(key,value) for key,value in data.items()]
    
    todo_game_files = [elem[-1] for elem in todo_files][:max_files] 
    
    env_start_time = time.time()
    print('Loading Environments...')
    
    # 优化并行环境构建 - 每次只处理一个游戏文件
    # 先处理第一个文件以初始化环境
    first_game_file = todo_game_files[0]
    parallel_env = build_parallel_alfworld_envs(gamefiles=[first_game_file],
                                                group_n=1,
                                                resources_per_worker={'num_cpus': 0.5},  # 增加每个worker的CPU资源
                                                num_parallel=num_parallel,
                                                num_copied=0,
                                                is_train=False) 
    
    print("-"*50 + "build parallel env done" + "-"*50)
    
    TASK_TYPES = {
        "pick": "pick_and_place_simple",
        "look": "look_at_obj_in_light",
        "clean": "pick_clean_then_place_in_recep",
        "heat": "pick_heat_then_place_in_recep",
        "cook": "pick_cool_then_place_in_recep",
        "pick2": "pick_two_obj_and_place"
    } 
    
    # 优化采样参数·
    sampling_params = SamplingParams(n=1, max_tokens=512, temperature=0.)  # 进一步减少最大token数
    
    correct_count = 0
    
    conversations = [] 
    
    print("-"*50 + "starting loop" + "-"*50)
    # 逐个处理游戏文件，避免同时加载过多环境
    for idx, game_file in enumerate(tqdm(todo_game_files)):
        print(f"Processing game file {idx+1}/{len(todo_game_files)}")
        
        # 为每个游戏文件重新创建环境，避免累积进程
        if idx > 0:
            # 关闭之前的环境
            parallel_env.close()
            # 创建新的环境
            parallel_env = build_parallel_alfworld_envs(gamefiles=[game_file],
                                                        group_n=1,
                                                        resources_per_worker={'num_cpus': 0.5},
                                                        num_parallel=num_parallel,
                                                        num_copied=0,
                                                        is_train=False)
        
        save_dict = {} 
        save_dict['gamefile'] = game_file 
        for task_key, task_name in TASK_TYPES.items():
            if task_name in game_file:
                save_dict['task_type'] = task_name
                break
        
        obs, possible_actions = parallel_env.get_start_info_file(game_file) 
        
        obv, task = obs.split('\n\nYour task is to: ')
        
        initial_prompt = compressed_prompt_initial.format(task_description=task,
                                                      current_observation=obv,
                                                      admissible_actions=possible_actions)
        start_completion = [
            {'role':'system','content':system_prompt},
            {'role':'user','content':initial_prompt + limit_prompt}
        ]
        
        start_prompt = tokenizer.apply_chat_template(start_completion,
                                                 add_generation_prompt=True,
                                                 tokenize=False) 
        
        save_dict['conversation'] = []
        save_dict['conversation'].extend(start_completion) 
        
        action_manager = {}
        obs_manager = {}
        success = False
        generation_prompt = start_prompt 
        
        # 限制最大回合数
        max_turns = 20
        for i in range(max_turns): 
            turn_dict = {} 
            vllm_response = model.generate([generation_prompt + '<think>'], sampling_params)
            model_response = parse_vllm_output(vllm_response) 
            actions_w_think = extract_think_and_actions(model_response[0]) 
            actions = actions_w_think['actions'] 
            
            if len(actions) >= num_parallel:
                success = False
                break
            
            feedback = parallel_env.step_file(game_file, actions) 
            observations = feedback[0]
            scores_list = feedback[1]
            total_obv = feedback[-1] 
            
            turn_dict['actions'] = actions
            turn_dict['observations'] = observations
            turn_dict['parallel_num'] = len(observations) 
            turn_dict['turn'] = i 
            save_dict['conversation'].append(turn_dict)
            
            for score in scores_list: 
                if score == 1:
                    success = True 
            
            if success:
                correct_count += 1
                break 
            
            admissible_commands = [elem['admissible_commands'] for elem in feedback[3]]
            
            store_idx = 0 
            for key, value in actions.items():
                if key not in action_manager:
                    action_manager[key] = [value] 
                else:
                    action_manager[key].append(value)
                
                if key not in obs_manager:
                    obs_manager[key] = [observations[store_idx]]
                else:
                    obs_manager[key].append(observations[store_idx])
                store_idx += 1
            
            last_action_obv = ''
            idx = 0
            for env_idx, action in actions.items():
                action_cur_env = action
                obs_cur_env = observations[idx] 
                poa = admissible_commands[idx]
                idx += 1 
                last_action_obv += f'In Environment {env_idx}\n'
                last_action_obv += f'Action: {action_cur_env}\n'
                last_action_obv += f'Observation: {obs_cur_env}\n'
                last_action_obv += f'Admissible Actions: {poa}\n'
            
            if i == 0:
                history_prompt = ''
            
            else:
                history_prompt = ''
                # 只保留最近的1轮历史，进一步减少提示长度
                max_history = 1
                for env_idx in action_manager.keys():
                    action_cur_env = action_manager[env_idx][-max_history:]
                    obs_cur_env = obs_manager[env_idx][-max_history:]
                    history_prompt += f'In Environment {env_idx}\n'
                    for history_idx, (action, z_obs) in enumerate(zip(action_cur_env, obs_cur_env)):
                        history_prompt += f'Action {len(action_manager[env_idx]) - max_history + history_idx + 1}: {action}\n'
                        history_prompt += f'Observation {len(obs_manager[env_idx]) - max_history + history_idx + 1}: {z_obs}\n'
                
                history_prompt = f'\nYou have already taken multiple actions in multiple parallel environments. Below are the most recent observaitons and the corresponding actions you took:\n{history_prompt}\n'
            
            generation_prompt = compressed_prompt_process.format(task_description=task,
                                                         initial_observation=obv,
                                                         history_info=history_prompt,
                                                         last_history=last_action_obv) 
            
            generation_completion = [
                {'role':'system','content':system_prompt},
                {'role':'user','content':generation_prompt + limit_prompt}
            ]
            
            generation_prompt = tokenizer.apply_chat_template(generation_completion,
                                                              add_generation_prompt=True,
                                                              tokenize=False) 
            
        save_dict['turn'] = i 
        save_dict['success'] = success 
        conversations.append(save_dict) 
        
        # 清理内存
        del action_manager
        del obs_manager
        torch.cuda.empty_cache()
        
        # 每处理3个文件后休息一下，让系统有时间清理资源
        if (idx + 1) % 3 == 0:
            print("Taking a short break to free up resources...")
            time.sleep(3)
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    name = 'evaluate_light'
    step = '1'
    with open(f'{save_path}/{name}_{step}.json'.replace('baselines','evaluate'), 'w') as f:
        json.dump(conversations, f, indent=4) 
    
    # 关闭环境
    parallel_env.close()
    ray.shutdown()
    
    print(f"Evaluation completed. Correct count: {correct_count}")

if __name__ == "__main__":
    # print("import done")
    eval()