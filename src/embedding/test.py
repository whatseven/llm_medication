import requests
import json
import sys
import os

# 添加embedding模块路径
sys.path.append(os.path.dirname(__file__))
from embedding import get_embedding

def test_api_directly():
    """直接测试API调用（使用官网示例的模型）"""
    print("=== 测试1: 直接API调用（官网示例模型） ===")
    
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
    }
    headers = {
        "Authorization": "Bearer sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        print(f"API响应状态: {response.status_code}")
        print(f"响应数据结构: {list(result.keys())}")
        
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            print(f"向量维度: {len(embedding)}")
            print(f"向量前5个元素: {embedding[:5]}")
            print("✅ 官网示例模型调用成功")
            return True
        else:
            print(f"❌ API响应格式异常: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 直接API调用失败: {e}")
        return False

def test_qwen_model():
    """测试Qwen模型"""
    print("\n=== 测试2: Qwen模型测试 ===")
    
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": "胸痛 呼吸困难 乏力"
    }
    headers = {
        "Authorization": "Bearer sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Qwen模型响应状态: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"Qwen模型向量维度: {len(embedding)}")
                print(f"向量前5个元素: {embedding[:5]}")
                print("✅ Qwen模型调用成功")
                return True
        
        print("❌ Qwen模型调用失败")
        return False
        
    except Exception as e:
        print(f"❌ Qwen模型测试失败: {e}")
        return False

def test_embedding_function():
    """测试embedding.py中的函数"""
    print("\n=== 测试3: embedding.py函数测试 ===")
    
    api_token = "sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns"
    
    # 测试症状向量化
    symptoms_texts = [
        "胸痛 呼吸困难 乏力",
        "头痛 发热 咳嗽",
        "腹痛 恶心 呕吐"
    ]
    
    for i, text in enumerate(symptoms_texts, 1):
        print(f"\n测试症状 {i}: {text}")
        embedding = get_embedding(text, api_token)
        
        if embedding and len(embedding) > 0:
            print(f"  ✅ 向量化成功，维度: {len(embedding)}")
            print(f"  向量前5个元素: {embedding[:5]}")
            
            # 检查是否为全零向量
            if all(x == 0.0 for x in embedding):
                print(f"  ⚠️ 警告: 这是一个全零向量!")
            else:
                print(f"  ✅ 向量正常（非全零）")
        else:
            print(f"  ❌ 向量化失败")

def test_symptom_data():
    """测试真实症状数据"""
    print("\n=== 测试4: 真实症状数据测试 ===")
    
    # 加载测试数据
    test_data_path = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/milvus_data/test.json"
    
    # 创建测试数据文件（如果不存在）
    test_data = [
        {
            "_id": {"$oid": "test001"},
            "name": "测试疾病1",
            "symptom": ["胸痛", "呼吸困难", "乏力"]
        },
        {
            "_id": {"$oid": "test002"},
            "name": "测试疾病2", 
            "symptom": ["头痛", "发热", "咳嗽"]
        }
    ]
    
    # 确保目录存在
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试数据已保存到: {test_data_path}")
    
    # 测试症状向量化
    api_token = "sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns"
    
    for record in test_data:
        symptoms = record["symptom"]
        symptoms_text = " ".join(symptoms)
        
        print(f"\n疾病: {record['name']}")
        print(f"症状: {symptoms}")
        print(f"合并文本: {symptoms_text}")
        
        embedding = get_embedding(symptoms_text, api_token)
        
        if embedding and len(embedding) > 0:
            is_zero_vector = all(x == 0.0 for x in embedding)
            print(f"  向量维度: {len(embedding)}")
            print(f"  是否为全零向量: {is_zero_vector}")
            if not is_zero_vector:
                print(f"  向量前5个元素: {embedding[:5]}")
                print("  ✅ 向量化成功")
            else:
                print("  ❌ 返回全零向量")
        else:
            print("  ❌ 向量化失败")

if __name__ == "__main__":
    print("开始测试Embedding API功能...\n")
    
    # 运行所有测试
    test_api_directly()
    test_qwen_model()
    test_embedding_function()
    test_symptom_data()
    
    print("\n=== 测试完成 ===")
