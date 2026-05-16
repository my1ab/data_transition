# Base Import
import os 
import json  
import ray
import yaml 
import time
import torch
import tempfile 
import numpy as np
from os.path import join as pjoin
import torchvision.transforms as T
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Any 

from tqdm import tqdm
from prompts import system_prompt,reason_prompt 
from utils import deepseek,extract_think_and_actions 

# ALF-World Import
import gymnasium as gym
from gymnasium import spaces

import textworld 
import textworld.gym 
from alfworld.info import ALFWORLD_DATA 
from agent_system.environments.prompts import *
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment 

import re
from openai import OpenAI
from copy import deepcopy

def get_env_name(game_file):
    return game_file.split('json_2.1.1/train/')[-1].replace('/game.tw-pddl','') 

def deepseek(messages):
    client = OpenAI(api_key="Your DeepSeek API Here", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )
    
    return response.choices[0].message.content 


def extract_think_and_actions(text):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None
    
    actions_pattern = r'<env_\d+>(.*?)</env_\d+>'
    actions = re.findall(actions_pattern, text, re.DOTALL)
    actions_dict = {}
    for index,action in enumerate(actions):
        actions_dict[index + 1] = action
    # actions = [{index+1:action} ]
    
    return {
        'think': think_content,
        'actions': actions_dict
    }

def read_json(file_path):
    data = json.load(open(file_path,'r'))
    return data

# Set the TMPDIR avoiding the fking `No Space Left On Device`
os.environ['TMPDIR'] = '/diskpool/tmp'   
tempfile.tempdir = '/diskpool/tmp'

# Base Env
class Env: 
    def __init__(self,game_file):
        self.gamefile = game_file 
        env,obs,infos = self.build_env(gamefile=game_file)  # Build the Environment
        # Initialize some class attributes
        self.env = env
        self.start_obv = obs  # Record the start observation
        self.start_infos = infos  # Record the start infos
        self.last_command = [] # Record the actions in last step
        self.auto_reset = True
        self.is_done = False
    
    # Execute a action in this environment
    def step(self,action):
        if self.is_done is True:
            obs, reward, done, infos = self.last_command[-1]

            if self.auto_reset:
                reward, done = 0., False
                obs, infos = self.reset() 
        else:
            obs, rewards, dones, infos = self.env.step(action) 
            # `obs` means the result of executing `action` in enviroment
            # There are `admissible_commands` in infos which means the actions that agent can take in next step.
            # 

        if dones:
            self.is_done = True 

        self.last_command.append(
            {
                'action':action,
                'observation':obs,
                'rewards':rewards,
                'dones':dones,
                'possible_commands':infos['admissible_commands'],
                'game_file':infos['admissible_commands']
            }
        )

        return obs, rewards, dones, infos
    
    # Reset the status of current env
    def reset(self):
        obs, infos = self.env.reset() 
        return obs, infos
    
    # Build the environment with gamefile
    def build_env(self,gamefile):
        # Don't Need to figure out the code here, just know what can it do.
        request_infos = textworld.EnvInfos(facts=True,admissible_commands=True,extras=["gamefile"])

        env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(),])
        
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()

        return env,obs,infos


# This class is implemented for parallel agent that can explore multiple parallel
# Multiple paralllel environments serve for a single Agent
class ParallelAlfworldWorker:
    def __init__(self, game_files, num_parallel, num_copied): 
        # For Saving Parallel Environments
        self.env_pools = {} 
        # Initialize 
        for parallel_idx in range(num_parallel): 
            self.env_pools[parallel_idx + 1] = Env(game_files) if num_copied == 0 else [Env(game_files) for _ in range(num_copied)] 
        
        # Record the start `observations` and `possible commands` in next step.
        self.start_obv = self.env_pools[1].start_obv 
        self.admissible_commands = self.env_pools[1].start_infos['admissible_commands'] 

        # self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        # self.env.seed(seed)

    def show_basis_infos(self):
        return self.start_obv,self.admissible_commands
    
    # Execute parallel actions in parallel Environments
    def step(self, action_dict): 
        obs,scores,dones,infos = [],[],[],[]
        obs_prompt = '' 
        for action_indx,action in action_dict.items():
            sub_env = self.env_pools[action_indx]
            ob,reward,done,info = sub_env.step(action)

            admissible_commands = ','.join(info['admissible_commands']) 

            obs_prompt += f'<observation_{action_indx}>\nThe observation and next candidated actions of {action_indx}-th environment are:\nObservation:\n{ob}\nNext Possible Actions:\n{admissible_commands}\n</observation_{action_indx}>\n'

            obs.append(ob) 
            scores.append(reward)
            dones.append(done)
            infos.append(info)
    
        return obs, scores, dones, infos, obs_prompt
    
    # Reset
    def reset(self):
        """Reset the environment"""

        for env in self.env_pools.values():
            obs, infos = env.reset() 
        
        return obs, infos

# For a single task, sample a group of answers, each group has `group_n` answer
# This class is implemented for `GRPO` Algorithms or some situations requiring sample multiple answer
class ParallelAlfworldEnvs(gym.Env):
    def __init__(self, 
                 game_files,
                 group_n, 
                 resources_per_worker, 
                 is_train=True, 
                 num_parallel=10,
                 num_copied=0,
                 env_kwargs={}):
        super().__init__() 
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.multi_modal = False
        # self.num_processes = env_num * group_n
        self.group_n = group_n
        
        # Create Ray remote actors instead of processes 
        env_worker = ray.remote(**resources_per_worker)(ParallelAlfworldWorker)
        self.workers = [] 
        self.workers_dict = {} 

        for game_file in game_files: 
            worker = env_worker.remote(game_file,num_parallel, num_copied)
            self.workers.append(worker) 
            self.workers_dict[get_env_name(game_file)] = worker
    
    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = [] 
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i]) 
            futures.append(future) 
        
        # Collect results
        observation_list = []
        scores_list = []
        dones_list = []
        infos_list = []
        obs_prompt_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, infos,prompts) in enumerate(results):
            observation_list.append(obs)
            scores_list.append(scores)
            dones_list.append(dones)
            infos_list.append(infos)
            obs_prompt_list.append(prompts)
        
        return observation_list, scores_list, dones_list, infos_list, obs_prompt_list
    
    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)
        
        obs = []
        infos = [] 
        results = ray.get(futures)
        for obv,info in results:
            obs.append(obv)
            infos.append(info)

        return obs, infos
    
    def step_file(self,game_file,action):
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 
        future = worker.step.remote(action) 
        results = ray.get(future)
        # results = future.results() 
        return results[0], results[1], results[2], results[3], results[4] 
    
    def get_start_info_file(self,game_file):
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 
        future = worker.show_basis_infos.remote()
        results = ray.get(future)

        return results[0],results[1] # obv,infos

    def reset_file(self,game_file):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        sub_gamefile = get_env_name(game_file)
        worker = self.workers_dict[sub_gamefile] 

        future = worker.reset.remote()
        result = ray.get(future)
        
        return result[0], result[1]
    
    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands 

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_parallel_alfworld_envs(gamefiles,
                                #  env_num, 
                                 group_n, 
                                 resources_per_worker, 
                                 num_parallel,
                                 num_copied,
                                 is_train=True, 
                                 env_kwargs={}):
    return ParallelAlfworldEnvs(gamefiles,
                                # env_num, 
                                group_n, 
                                resources_per_worker, 
                                is_train,
                                num_parallel=num_parallel,
                                num_copied=num_copied) 

game_path = '/dir_path/gamefiles_train.json'

data = read_json(game_path) 
path = 'Save Path Here'

exist_files = [elem.replace('.json','') for elem in os.listdir(path)]


todo_files = [(key,value) for key,value in data.items() if str(key) not in set(exist_files)][:5] 
todo_game_files = [elem[-1] for elem in todo_files] 

import time

env_start_time = time.time()
print('Loading Environments...')
parallel_env = build_parallel_alfworld_envs(gamefiles=todo_game_files,
                                            # env_num=512,
                                            group_n=1,
                                            resources_per_worker={'num_cpus': 0.1},
                                            num_parallel=10,
                                            num_copied=0, # Useless Parameter
                                            is_train=True # Useless 2
                                            ) 

# Main Logic
def get_single_trajectory(parallel_env,game_file,turns=50,save_path=None):
    save_data_dict = {'game_file':game_file,} 
    obs,admissible_commands = parallel_env.get_start_info_file(game_file) 
    conversations = [
        {'role':'system','content':system_prompt},
        {
            'role':'user',
            'content': reason_prompt.format(
                current_observation=obs,
                admissible_actions=admissible_commands
                )
        }
    ] 
    success = False 
    for i in tqdm(range(turns)):
        # feed the llm and get the output
        output = deepseek(messages=conversations) 
        result = extract_think_and_actions(output) 
        
        conversations.append({'role':'assistant','content':output})
        if result['actions'] != {}:
            obs_prompt = {'role':'user'} 
            # run the fucking env 
            feedback = parallel_env.step_file(game_file,result['actions']) 
            
            obs_list = feedback[0]
            scores_list = feedback[1]
            dones_list = feedback[2]
            infos_list = feedback[3]
            observation_str = feedback[4] 

            obs_prompt['content'] = observation_str
            
            conversations.append(obs_prompt) 
            
            for score in scores_list: 
                if score == 1:
                    success = True
            
            if success:
                break 

        else:
            break

    save_data_dict['turn'] = i + 1
    save_data_dict['success'] = success
    save_data_dict['conversations'] = conversations
    

    with open(save_path,'w') as f:
        json.dump(save_data_dict,f,indent=4)
    
    return save_data_dict 

output = parallel_env.reset() 
print(f'Environments have been successfule loaded, take {time.time() - env_start_time} seconds') 

import concurrent.futures
from threading import Lock

def process_single_file(elem, path, parallel_env, turns=50):
    """处理单个文件的函数"""
    key, value = elem
    save_path = f'{path}/{str(key)}.json'
    output = get_single_trajectory(parallel_env=parallel_env,
                                  game_file=value,
                                  turns=turns,
                                  save_path=save_path)
    return output

def process_files_multithreaded(todo_files, path, parallel_env, turns=50, max_workers=None):
    """多线程处理文件"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_elem = {
            executor.submit(process_single_file, elem, path, parallel_env, turns): elem 
            for elem in todo_files
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_elem):
            elem = future_to_elem[future] 
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'处理文件 {elem} 时发生错误: {exc}')
    
    return results 

results = process_files_multithreaded(todo_files, path, parallel_env, turns=50,max_workers=16) 