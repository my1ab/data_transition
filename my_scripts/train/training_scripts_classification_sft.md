# SFT/正式训练脚本分类（按序列长度规则）

## 分类规则

根据提供的序列长度配置规则：
- **SFT阶段**：最大序列长度 8192 tokens
- **正式训练**：较小的序列长度配置

## SFT训练脚本（Prompt=8192, Response=1024）

### alf-world/

| 脚本名称 | 特点 |
|---------|------|
| `grpo_alfworld_parallel_demo.sh` | 演示/调试脚本，使用演示数据 |

**特征：**
- `data.max_prompt_length=8192`
- `data.max_response_length=1024`
- 使用演示数据 `parallel_train_data_demo_easy.parquet`
- 训练轮数：1 epoch

**模型路径配置：**

| 脚本名称 | resume_from_path | model.path | default_local_dir |
|---------|-----------------|------------|------------------|
| `grpo_alfworld_parallel_demo.sh` | 无 | `OpenRLHF/ckpt/hf_ckpt_1.5b/global_step5871_hf` | `verl-agent/checkpoints/dummy` |

## 正式训练脚本（Prompt=2048, Response=512）

### alf-world/

| 脚本名称 | 特点 |
|---------|------|
| `grpo_alf_world.sh` | GRPO/GiGPO正式训练，150 epoch，支持World Size 2/4 |
| `grpo_alf_world_grpo.sh` | GRPO正式训练，150 epoch，4 GPU配置 |

**特征：**
- `data.max_prompt_length=2048`
- `data.max_response_length=512`
- 使用完整训练数据 `train.parquet`
- 训练轮数：150 epoch
- 入口函数：`main_ppo`

**模型路径配置：**

| 脚本名称 | model.path |
|---------|------------|
| `grpo_alf_world.sh` | `models/Qwen2.5-0.5B-Instruct` 或 `models/Qwen2.5-1.5B-Instruct` |
| `grpo_alf_world_grpo.sh` | `models/Qwen2.5-1.5B-Instruct` |

## 辅助脚本

### alf-world/

| 脚本名称 | 用途 |
|---------|------|
| `merge_model.sh` | 合并FSDP训练后的模型权重 |
| `resume.sh` | 断点恢复参数模板 |

**模型路径配置：**

| 脚本名称 | resume_from_path |
|---------|-----------------|
| `merge_model.sh` | 无 |
| `resume.sh` | `verl-agent/checkpoints/grpo_1.5_coldstart_epoch1_outcome/global_step_75/actor` |

## 奖励模型训练脚本

**未发现专门的奖励模型训练脚本**。脚本名称中的 `reward_lora` 表示使用奖励模型进行PPO训练，而非训练奖励模型本身。

## 分类汇总

| 类型 | 数量 | 说明 |
|-----|------|------|
| RL训练 | 12个 | Prompt=10240, Response=2048 |
| SFT训练 | 1个 | Prompt=8192, Response=1024 |
| 正式训练 | 2个 | Prompt=2048, Response=512 |
| 辅助脚本 | 2个 | 模型合并和断点恢复 |
| 奖励模型训练 | 0个 | 无专门脚本 |

## 注意事项

1. **文件位置问题**：alf-world/3b/目录下存在5个文件名包含`1.5b`的脚本，可能是命名或目录组织错误
2. **epoch实现**：部分脚本通过设置`trainer.resume_mode=auto`和`trainer.resume_from_path`实现多epoch训练，实际`trainer.total_epochs`可能为1