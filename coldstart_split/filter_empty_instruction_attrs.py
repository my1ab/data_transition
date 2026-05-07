# 删除人类标注中没有 instruction_attributes 属性的标注，保持格式一致
import json
import os
from tqdm import tqdm

def filter_empty_instruction_attrs(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 - {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = {}
    total_annotations = 0
    removed_annotations = 0
    kept_annotations = 0
    
    for asin, annotations in tqdm(data.items(), desc="Filtering annotations", total=len(data)):
        if isinstance(annotations, list):
            valid_annotations = []
            for annotation in annotations:
                total_annotations += 1
                if 'instruction_attributes' in annotation and annotation['instruction_attributes']:
                    valid_annotations.append(annotation)
                    kept_annotations += 1
                else:
                    removed_annotations += 1
            
            if valid_annotations:
                filtered_data[asin] = valid_annotations
        else:
            total_annotations += 1
            if 'instruction_attributes' in annotations and annotations['instruction_attributes']:
                filtered_data[asin] = annotations
                kept_annotations += 1
            else:
                removed_annotations += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("过滤结果报告")
    print("=" * 60)
    print(f"原始商品数量: {len(data)}")
    print(f"过滤后商品数量: {len(filtered_data)}")
    print(f"原始标注总数: {total_annotations}")
    print(f"保留的标注数: {kept_annotations}")
    print(f"移除的标注数: {removed_annotations}")
    print(f"移除率: {(removed_annotations/total_annotations)*100:.2f}%")
    print(f"结果已保存到: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    input_path = '/home/dpepo/data/items_human_ins.json'
    output_path = '/home/dpepo/data/items_human_ins_cleaned.json'
    
    filter_empty_instruction_attrs(input_path, output_path)