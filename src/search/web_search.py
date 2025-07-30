import requests
import json
import os
from typing import Dict, Any, Optional


def web_search(query: str, 
               summary: bool = True, 
               count: int = 10,
               api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    使用博茶AI的web搜索API进行搜索
    
    Args:
        query (str): 搜索查询内容
        summary (bool): 是否返回摘要，默认True
        count (int): 返回结果数量，默认10
        api_key (str, optional): API密钥，如果不提供则从环境变量获取
    
    Returns:
        Dict[str, Any]: 搜索结果的JSON响应
    
    Raises:
        ValueError: 当API密钥缺失时
        requests.RequestException: 当API请求失败时
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.getenv('BOCHAAI_API_KEY')
        if api_key is None:
            raise ValueError("API密钥未提供，请设置环境变量BOCHAAI_API_KEY或直接传入api_key参数")
    
    url = "https://api.bochaai.com/v1/web-search"
    
    payload = json.dumps({
        "query": query,
        "summary": summary,
        "count": count
    })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # 如果状态码不是200会抛出异常
        return response.json()
    except requests.RequestException as e:
        print(f"Web搜索API请求失败: {str(e)}")
        raise


def search_medical_info(query: str, count: int = 5) -> Dict[str, Any]:
    """
    专门用于医疗信息搜索的便捷函数
    
    Args:
        query (str): 医疗相关的搜索查询
        count (int): 返回结果数量，默认5
    
    Returns:
        Dict[str, Any]: 搜索结果
    """
    # 在查询前添加医疗相关的关键词以提高搜索准确性
    medical_query = f"医疗 医学 {query}"
    return web_search(medical_query, summary=True, count=count)


if __name__ == "__main__":
    # 测试示例
    print("=== Web搜索功能测试 ===\n")
    
    # 测试1: 基础搜索功能
    try:
        print("测试1: 基础搜索 - 天空为什么是蓝色的？")
        result1 = web_search(
            query="天空为什么是蓝色的？",
            summary=True,
            count=3,
            api_key="sk-458cada7190d42e0aff052c288ab05c5"  # 请替换为您的真实API密钥
        )
        print("搜索结果格式:")
        print(json.dumps(result1, ensure_ascii=False, indent=2))
        print(f"结果类型: {type(result1)}")
        if isinstance(result1, dict):
            print(f"结果包含的键: {list(result1.keys())}")
        print("-" * 50)
        
    except Exception as e:
        print(f"测试1失败: {str(e)}")
        print("-" * 50)
    
    # 测试2: 医疗信息搜索
    try:
        print("测试2: 医疗信息搜索 - 高血压的症状")
        result2 = search_medical_info("高血压的症状", count=3)
        print("医疗搜索结果格式:")
        print(json.dumps(result2, ensure_ascii=False, indent=2))
        print("-" * 50)
        
    except Exception as e:
        print(f"测试2失败: {str(e)}")
        print("-" * 50)
    
    # 测试3: 检查返回数据的结构
    try:
        print("测试3: 分析返回数据结构")
        result3 = web_search("Python编程", summary=True, count=2, api_key="sk-458cada7190d42e0aff052c288ab05c5")
        
        print("数据结构分析:")
        print(f"- 根对象类型: {type(result3)}")
        
        if isinstance(result3, dict):
            for key, value in result3.items():
                print(f"- {key}: {type(value)}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"  └─ 列表元素类型: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"  └─ 列表元素包含的键: {list(value[0].keys())}")
        
    except Exception as e:
        print(f"测试3失败: {str(e)}")
    
    print("\n=== 测试完成 ===")
    print("注意: 请将api_key参数替换为您的真实API密钥才能正常使用")
