from huggingface_hub import snapshot_download
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR_NAME = MODEL_NAME.split("/")[-1]
OUTPUT_DIR = os.path.join("./model", MODEL_DIR_NAME)

print(f"正在下载 {MODEL_NAME} 模型...")
print(f"模型名称: {MODEL_NAME}")
print(f"输出目录: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False
    )
    print(f"\nQwen2.5-1.5B 模型下载完成！")
    print(f"模型已保存到: {os.path.abspath(OUTPUT_DIR)}")
except Exception as e:
    print(f"\n模型下载失败: {str(e)}")
    exit(1)