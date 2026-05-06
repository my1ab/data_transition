# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np
import os
import sys

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        # Get the path to this file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Add webshop directory for WebAgentTextEnv imports (web_agent_site is in this path)
        webshop_root = os.path.join(current_dir, 'webshop')
        if webshop_root not in sys.path:
            sys.path.append(webshop_root)
        web_agent_site_path = os.path.join(webshop_root, 'web_agent_site')
        if web_agent_site_path not in sys.path:
            sys.path.append(web_agent_site_path)
        
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        # 修改 消除了环境检查器的错误
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        # obs, reward, done, info = self.env.step(action)
        step_result = self.env.step(action)
        # 修改 兼容自定义的返回格式
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, info, *_ = step_result
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        split: str = 'train',
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            # Get the path to this file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Add project root (parent of agent_system) to PYTHONPATH for Ray workers
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
            
            # Get current PYTHONPATH
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if current_pythonpath:
                pythonpath_entries = current_pythonpath.split(os.pathsep)
            else:
                pythonpath_entries = []
            
            # Add project root if not already in PYTHONPATH
            if project_root not in pythonpath_entries:
                pythonpath_entries.append(project_root)
            
            # Create runtime_env to set PYTHONPATH for all Ray workers
            runtime_env = {
                'env_vars': {
                    'PYTHONPATH': os.pathsep.join(pythonpath_entries)
                }
            }
            
            # 启动ray之前添加路径
            ray.init(runtime_env=runtime_env)

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.split = split
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- Four-way split strategy (ordered by training flow) ----------#
        # sft: 0-2500      → SFT训练专用（训练第一步）
        # train: 2500-5000 → RL训练专用（训练第二步）
        # eval: 5000-6000  → RL验证专用（训练第三步）
        # test: 6000-      → 最终评估专用（训练第四步）
        # ---------------------------------------------------------------------------#
        # 按训练流程顺序排列：SFT → RL训练 → RL验证 → 最终评估
        if split == 'sft':
            self.goal_idxs = range(min(2500, len(goals)))
        elif split == 'train':
            self.goal_idxs = range(2500, min(5000, len(goals)))
        elif split == 'eval':
            self.goal_idxs = range(5000, min(6000, len(goals)))
        elif split == 'test':
            self.goal_idxs = range(6000, len(goals))
        else:
            # Default to train split if invalid split is provided
            self.goal_idxs = range(2500, min(5000, len(goals)))
            
        print(f"Split: {split}, Goal indices: {self.goal_idxs}")

        # 修改 original code before fix
        # if not self.is_train:
        #     self.goal_idxs = range(min(500, len(goals)))
        # # 进入分支
        # else:
        #     if len(goals) <= 500:
        #         self.goal_idxs = range(len(goals))
        #     else:
        #         self.goal_idxs = range(500, len(goals))
            
        # print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Original reset logic (before fix):
        # idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        # idx = np.repeat(idx, self.group_n).tolist()
        # 
        # Fixed reset logic (handles empty goal_idxs):
        goal_idxs_list = list(self.goal_idxs)
        if len(goal_idxs_list) == 0:
            raise ValueError("No goals available to sample from")
        idx = self._rng.choice(goal_idxs_list, size=min(self.env_num, len(goal_idxs_list)), replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean-up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    split: str = 'train',
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher-level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        split=split,
        env_kwargs=env_kwargs,
    )