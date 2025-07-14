import sys
import os
import json

# 添加search模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))
from milvus_search import search_similar_diseases

def test_search():
    """测试Milvus检索功能"""
    test_queries = [
        "胸痛呼吸困难",
        "头痛发热", 
        "腹痛恶心"
    ]
    
    print("=== Milvus检索测试 ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        print("-" * 40)
        
        results = search_similar_diseases(query)
    print(results)
    return results

if __name__ == "__main__":
    test_search()
