import sys
import os
import json

# Add search module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))
from milvus_search_copy_en import search_similar_diseases

def test_search_en():
    """Test Milvus search functionality for English data"""
    test_queries = [
        "chest pain difficulty breathing",
        "headache fever", 
        "abdominal pain nausea",
        "fatigue cough",
        "dizziness shortness of breath"
    ]
    
    print("=== Milvus English Search Test ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 40)
        
        results = search_similar_diseases(query)
        
        if results:
            print(f"Found {len(results)} results:")
            for j, result in enumerate(results, 1):
                print(f"\nResult {j}:")
                print(f"  Disease Name: {result['name']}")
                print(f"  Symptoms: {result['symptom']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
                print(f"  ID: {result['oid']}")
        else:
            print("❌ No results found")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_search_en()
