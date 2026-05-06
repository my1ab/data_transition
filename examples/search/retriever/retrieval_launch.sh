
# 数据存储根目录 faiss向量索引文件 语料库文件 语料库文件路径 文本名称+路径
save_path=$HOME/data/searchR1
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# 启动http服务器 加载索引+语料库 接收查询文本并返回最相似的文本片段
python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000 \