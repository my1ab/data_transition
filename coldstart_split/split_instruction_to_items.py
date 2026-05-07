# 将人类标注中每个 instruction 分离为独立的 item，并按采样分割为不同数据集
import json
import os
import random
from tqdm import tqdm

def split_instructions(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 - {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    split_data = {}
    total_instructions = 0
    
    for asin, annotations in tqdm(data.items(), desc="Splitting instructions", total=len(data)):
        if isinstance(annotations, list):
            for idx, annotation in enumerate(annotations):
                new_key = f"{asin}_{idx}"
                split_data[new_key] = annotation
                total_instructions += 1
        else:
            split_data[asin] = annotations
            total_instructions += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("分离结果报告")
    print("=" * 60)
    print(f"原始商品数量: {len(data)}")
    print(f"分离后独立 item 数量: {len(split_data)}")
    print(f"总 instruction 数量: {total_instructions}")
    print(f"结果已保存到: {output_path}")
    print("=" * 60)
    
    return split_data

def split_by_sampling(split_data, base_output_path, seed=42):
    random.seed(seed)
    
    items_list = list(split_data.items())
    total = len(items_list)
    
    head_1500 = dict(items_list[:1500])
    remaining_after_head = items_list[1500:]
    
    random.shuffle(remaining_after_head)
    sample_1012 = dict(remaining_after_head[:1012])
    exclude_1012 = dict(remaining_after_head[1012:])
    
    path_head_1500 = base_output_path.replace('.json', '_head_1500.json')
    path_1500_max = base_output_path.replace('.json', '_1500_max.json')
    path_sample_1012 = base_output_path.replace('.json', '_sample_1012.json')
    path_exclude_1012 = base_output_path.replace('.json', '_exclude_1012.json')
    
    with open(path_head_1500, 'w', encoding='utf-8') as f:
        json.dump(head_1500, f, indent=2, ensure_ascii=False)
    
    with open(path_1500_max, 'w', encoding='utf-8') as f:
        json.dump(dict(remaining_after_head), f, indent=2, ensure_ascii=False)
    
    with open(path_sample_1012, 'w', encoding='utf-8') as f:
        json.dump(sample_1012, f, indent=2, ensure_ascii=False)
    
    with open(path_exclude_1012, 'w', encoding='utf-8') as f:
        json.dump(exclude_1012, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("采样分割结果报告")
    print("=" * 60)
    print(f"总 item 数量: {total}")
    print(f"head_1500: {len(head_1500)} 条")
    print(f"1500_max (剩余部分): {len(remaining_after_head)} 条")
    print(f"sample_1012: {len(sample_1012)} 条")
    print(f"exclude_1012: {len(exclude_1012)} 条")
    print(f"seed: {seed}")
    print("\n输出文件:")
    print(f"  - {path_head_1500}")
    print(f"  - {path_1500_max}")
    print(f"  - {path_sample_1012}")
    print(f"  - {path_exclude_1012}")
    print("=" * 60)

if __name__ == "__main__":
    input_path = '/home/dpepo/data/items_human_ins_cleaned.json'
    output_path = '/home/dpepo/data/items_human_ins_split.json'
    
    split_data = split_instructions(input_path, output_path)
    if split_data:
        split_by_sampling(split_data, output_path, seed=42)