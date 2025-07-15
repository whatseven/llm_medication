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
        
        if results:
            print(f"找到 {len(results)} 条结果:")
            for j, result in enumerate(results, 1):
                print(f"\n结果 {j}:")
                print(f"  疾病名称: {result['name']}")
                print(f"  症状: {result['symptom']}")
                print(f"  相似度: {result['similarity_score']:.4f}")
                print(f"  ID: {result['oid']}")
        else:
            print("❌ 未找到结果")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_search()
