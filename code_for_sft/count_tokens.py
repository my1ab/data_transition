import json
from transformers import AutoTokenizer

def count_tokens_in_json(json_file_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_tokens = 0
    total_characters = 0
    item_count = len(data)
    
    for idx, item in enumerate(data):
        if 'messages' in item:
            messages = item['messages']
        elif 'input' in item:
            messages = item['input']
        else:
            continue
        
        for message in messages:
            content = message.get('content', '')
            total_characters += len(content)
            tokens = tokenizer.encode(content, add_special_tokens=False)
            total_tokens += len(tokens)
    
    return {
        'total_items': item_count,
        'total_characters': total_characters,
        'total_tokens': total_tokens,
        'tokens_per_character': total_tokens / total_characters if total_characters > 0 else 0
    }

if __name__ == '__main__':
    json_file = '/home/dpepo/verl-agent/code_for_sft/converted_output.json'
    model_path = '/home/dpepo/verl-agent/model/Qwen2.5-0.5B-Instruct'
    
    result = count_tokens_in_json(json_file, model_path)
    
    print("Token 统计结果:")
    print(f"数据条目数: {result['total_items']}")
    print(f"总字符数: {result['total_characters']:,}")
    print(f"总 Token 数: {result['total_tokens']:,}")
    print(f"字符/Token 比率: {result['tokens_per_character']:.4f}")