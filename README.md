1、目录介绍：
langgraph_rag 是用langgraph重新包装的代码

         执行入口 main.py
         
         python main.py --input datasets/query_social.json --output datasets/query_social_langgraph_pred.json  这是实验的整个流程
         main.py本质调用的是hybrid_rag_query.py，要提前建好向量库和知识库
         使用的是本地模型/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct
        

datasets 数据集目录
         两个数据集分别是datasets\query_social.json  #这个是lawbench数据集
                       datasets\训练数据_基础社会法_600条.json #这个是SocialLawQA数据集

         实验结果以及对比实验结果分别保存在datasets的lawBench和ocialLawQA目录下


compareExperi 保存对比实验脚本

eval\pinggu_compatible.py 是评估脚本

eval\output  保存评估的结果

vectorDB\social_law_milvus.db  #向量库

KG_output\social_law_7B_processed  #最终建好的知识图谱

还需要 mysql 数据库保存节点和关系信息



2、以下是构建法律知识图谱的脚本
```bash
# 1. 分块
python law_data_chunk.py

# 2. 提取（GraphRAG）
python law_extract_graphrag.py

# 3. 去重
python law_deal_triple.py

# 4. 构建图谱
python build_law_graph.py
    

# 5. 查询
python query_law_graph_vllm.py
```






































