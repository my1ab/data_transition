# RL训练脚本分类（按序列长度规则）

## 分类规则

根据提供的序列长度配置规则：
- **RL阶段**：最大Prompt长度 10240 tokens，最大Response长度 2048 tokens
- **SFT阶段**：最大序列长度 8192 tokens

## RL训练脚本（Prompt=10240, Response=2048）

### alf-world/1.5b/

| 脚本名称 | 奖励类型 |
|---------|---------|
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_1epoch.sh` | Outcome Reward |
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_3epoch.sh` | Outcome Reward |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_1epoch.sh` | Parallel Reward |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_3epoch.sh` | Parallel Reward |
| `grpo_alfworld_parallel_1.5b_500_process_reward_lora_1epoch.sh` | Process Reward |

**模型路径配置：**

| 脚本名称 | resume_from_path | model.path | default_local_dir |
|---------|-----------------|------------|------------------|
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_1epoch.sh` | 无 | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome` |
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_3epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome/global_step_75/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step5871_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome` |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_1epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome_parallel/global_step_15/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome_parallel` |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_3epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch3_outcome_parallel/global_step_60/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step5871_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch3_outcome_parallel` |
| `grpo_alfworld_parallel_1.5b_500_process_reward_lora_1epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_process/global_step_100/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_process` |

### alf-world/3b/

| 脚本名称 | 奖励类型 |
|---------|---------|
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_1epoch.sh` | Outcome Reward |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_1epoch.sh` | Parallel Reward |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_3epoch.sh` | Parallel Reward |
| `grpo_alfworld_parallel_1.5b_500_process_reward_lora_1epoch.sh` | Process Reward |
| `grpo_alfworld_parallel_3b_500_outcome_reward_lora_3epoch.sh` | Outcome Reward |
| `grpo_alfworld_parallel_3b_500_outcome_reward_lora_4epoch.sh` | Outcome Reward |

**模型路径配置：**

| 脚本名称 | resume_from_path | model.path | default_local_dir |
|---------|-----------------|------------|------------------|
| `grpo_alfworld_parallel_1.5b_500_outcome_reward_lora_1epoch.sh` | 无 | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome` |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_1epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome_parallel/global_step_15/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome_parallel` |
| `grpo_alfworld_parallel_1.5b_500_outcome_parallel_reward_lora_3epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch3_outcome_parallel/global_step_60/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step5871_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch3_outcome_parallel` |
| `grpo_alfworld_parallel_1.5b_500_process_reward_lora_1epoch.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_process/global_step_100/actor` | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step1957_hf` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_process` |
| `grpo_alfworld_parallel_3b_500_outcome_reward_lora_3epoch.sh` | `verl-agent/checkpoints/grpo_3_coldstart_epoch3_outcome_parallel/global_step_65/actor` | `OpenRLHF/ckpt/hf_ckpt_3b/global_step5871_hf` | `verl-agent/checkpoints/grpo_3_coldstart_epoch3_outcome_parallel` |
| `grpo_alfworld_parallel_3b_500_outcome_reward_lora_4epoch.sh` | `verl-agent/checkpoints/grpo_3b_coldstart_epoch4_outcome_parallel/global_step_85/actor` | `OpenRLHF/ckpt/hf_ckpt_3b/global_step7828_hf` | `verl-agent/checkpoints/grpo_3b_coldstart_epoch4_outcome_parallel` |

### science-world/7b/

| 脚本名称 | 奖励类型 |
|---------|---------|
| `grpo_alfworld_parallel_7b_500_outcome_reward_lora_3epoch.sh` | Outcome Reward |

**模型路径配置：**

| 脚本名称 | resume_from_path | model.path | default_local_dir |
|---------|-----------------|------------|------------------|
| `grpo_alfworld_parallel_7b_500_outcome_reward_lora_3epoch.sh` | 无 | `Parallel-Qwen2.5-7B-Instruct-ColdStart-Epoch3-SciWorld` | `verl-agent/checkpoints/grpo_7b_coldstart_epoch3_outcome` |

## RL训练脚本特征

| 特征 | 配置值 |
|-----|-------|
| `data.max_prompt_length` | 10240 tokens |
| `data.max_response_length` | 2048 tokens |
| 入口函数 | `main_ppo_alfworld` 或 `main_ppo_sciworld` |
| 微调方式 | LoRA（`lora_rank=32`, `lora_alpha=32`） |
| 数据规模 | 500条并行训练数据 |

## 总结

**RL训练脚本总数：12个**
- alf-world/1.5b：5个
- alf-world/3b：6个
- science-world/7b：1个