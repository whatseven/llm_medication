import sys
import os

# 添加search模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))
from neo4j_search import neo4j_disease_search

def test_neo4j_disease_search():
    """测试Neo4j疾病搜索功能"""
    test_diseases = [
        "Pulmonary Aspergillosis",
        "肺炎", 
        "高血压"
    ]
    
    print("=== Neo4j疾病搜索测试 ===\n")
    
    for i, disease in enumerate(test_diseases, 1):
        print(f"测试 {i}: {disease}")
        print("-" * 50)
        
        result = neo4j_disease_search(disease)
        
        if result:
            print(result)
        else:
            print(f"❌ 未找到疾病 '{disease}' 的信息")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_neo4j_disease_search()
