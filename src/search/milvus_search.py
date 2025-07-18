import sys
import os
from typing import List, Dict, Any
from pymilvus import connections, db, Collection

# 添加embedding模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

def search_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    在Milvus中搜索相似疾病
    
    Args:
        query: 用户查询文本
        top_k: 返回前k个结果，默认3
        
    Returns:
        List[Dict]: 包含疾病信息的字典列表，不包含向量字段
    """
    # 配置信息
    host = "localhost"
    port = "19530"
    api_token = "sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns"
    database_name = "llm_medication"
    collection_name = "medication"
    partition_name = "knowledge_base"
    dimension = 4096
    
    try:
        # 连接Milvus
        connections.connect("default", host=host, port=port)
        db.using_database(database_name)
        
        # 获取collection
        collection = Collection(collection_name)
        
        # 向量化查询
        query_vector = get_embedding(query, api_token)
        if not query_vector or len(query_vector) != dimension:
            return []
        
        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        # 执行搜索
        results = collection.search(
            data=[query_vector],
            anns_field="symptom_vector",
            param=search_params,
            limit=top_k,
            output_fields=["oid", "name", "desc", "symptom"],
            partition_names=[partition_name]
        )
        
        # 处理结果
        if not results or len(results[0]) == 0:
            return []
        
        search_results = []
        for hit in results[0]:
            result_dict = {
                'oid': hit.entity.get('oid'),
                'name': hit.entity.get('name'),
                'desc': hit.entity.get('desc'),
                'symptom': hit.entity.get('symptom'),
                'similarity_score': float(hit.distance)
            }
            search_results.append(result_dict)
        
        return search_results
        
    except Exception as e:
        print(f"搜索错误: {e}")
        return []
