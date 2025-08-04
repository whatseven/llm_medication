import sys
import os

# 添加search模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))
from neo4j_diagnose_en import neo4j_diagnosis_search

def test_neo4j_diagnosis_search_en():
    """Test Neo4j English disease diagnosis search functionality"""
    test_diseases = [
        "Pulmonary Alveolar Proteinosis",
        "Hypertension",
        "Pneumonia",
        "Diabetes Mellitus",
        "Asthma"
    ]
    
    print("=== Neo4j English Disease Diagnosis Search Test ===\n")
    
    for i, disease in enumerate(test_diseases, 1):
        print(f"Test {i}: {disease}")
        print("-" * 50)
        
        result = neo4j_diagnosis_search(disease)
        
        if result:
            print(result)
        else:
            print(f"❌ No information found for disease '{disease}'")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_neo4j_diagnosis_search_en()
