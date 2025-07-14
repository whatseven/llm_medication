import sys
import os

# 添加search模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))
from neo4j_diagnose import neo4j_diagnosis_search

def test_neo4j_diagnosis():
    """测试Neo4j诊断搜索功能"""
    test_diseases = ["病毒性肝炎", "肺炎", "高血压"]
    
    print("=== Neo4j诊断信息测试 ===\n")
    
    for disease in test_diseases:
        print(f"测试疾病: {disease}")
        print("-" * 40)
        
        result = neo4j_diagnosis_search(disease)
        
        if result:
            print(result)
        else:
            print(f"❌ 未找到疾病 '{disease}' 的诊断信息")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_neo4j_diagnosis()
