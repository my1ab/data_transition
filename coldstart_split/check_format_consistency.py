# 检查过滤前后文件的格式一致性，确保数据结构完全相同
import json
import os

def check_format_consistency(original_path, filtered_path):
    if not os.path.exists(original_path):
        print(f"错误：原始文件不存在 - {original_path}")
        return
    
    if not os.path.exists(filtered_path):
        print(f"错误：过滤后文件不存在 - {filtered_path}")
        return
    
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(filtered_path, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    print("=" * 60)
    print("格式一致性检查报告")
    print("=" * 60)
    
    print(f"\n1. 数据类型检查:")
    print(f"   原始文件数据类型: {type(original_data).__name__}")
    print(f"   过滤后文件数据类型: {type(filtered_data).__name__}")
    
    if len(original_data) > 0:
        first_original_key = list(original_data.keys())[0]
        original_value_type = type(original_data[first_original_key]).__name__
        print(f"\n2. 原始文件第一个值的类型: {original_value_type}")
        
        if len(filtered_data) > 0:
            first_filtered_key = list(filtered_data.keys())[0]
            filtered_value_type = type(filtered_data[first_filtered_key]).__name__
            print(f"   过滤后文件第一个值的类型: {filtered_value_type}")
            
            if original_value_type == filtered_value_type:
                print(f"   ✓ 类型一致")
            else:
                print(f"   ✗ 类型不一致")
        
        original_sample = json.dumps(original_data[first_original_key], indent=2, ensure_ascii=False)
        print(f"\n3. 原始文件第一个商品结构:")
        print(f"   {first_original_key}:")
        print(original_sample)
        
        if len(filtered_data) > 0:
            first_filtered_key = list(filtered_data.keys())[0]
            filtered_sample = json.dumps(filtered_data[first_filtered_key], indent=2, ensure_ascii=False)
            print(f"\n4. 过滤后文件第一个商品结构:")
            print(f"   {first_filtered_key}:")
            print(filtered_sample)
        
        print(f"\n5. 字段对比:")
        original_item = original_data[first_original_key]
        if isinstance(original_item, list) and len(original_item) > 0:
            original_item = original_item[0]
        
        original_fields = set(original_item.keys())
        print(f"   原始文件包含字段: {original_fields}")
        
        if len(filtered_data) > 0:
            filtered_item = filtered_data[first_filtered_key]
            if isinstance(filtered_item, list) and len(filtered_item) > 0:
                filtered_item = filtered_item[0]
            
            filtered_fields = set(filtered_item.keys())
            print(f"   过滤后文件包含字段: {filtered_fields}")
            
            if original_fields == filtered_fields:
                print(f"   ✓ 字段完全一致")
            else:
                missing_fields = original_fields - filtered_fields
                extra_fields = filtered_fields - original_fields
                if missing_fields:
                    print(f"   ✗ 缺失字段: {missing_fields}")
                if extra_fields:
                    print(f"   ✗ 多余字段: {extra_fields}")
    
    print(f"\n6. 文件大小:")
    original_size = os.path.getsize(original_path)
    filtered_size = os.path.getsize(filtered_path)
    print(f"   原始文件大小: {original_size:,} bytes")
    print(f"   过滤后文件大小: {filtered_size:,} bytes")
    
    print(f"\n7. 商品数量:")
    print(f"   原始文件商品数量: {len(original_data)}")
    print(f"   过滤后文件商品数量: {len(filtered_data)}")
    
    print("\n" + "=" * 60)
    print("检查完成！")

if __name__ == "__main__":
    original_path = '/home/dpepo/data/items_human_ins.json'
    filtered_path = '/home/dpepo/data/items_human_ins_filtered.json'
    
    check_format_consistency(original_path, filtered_path)