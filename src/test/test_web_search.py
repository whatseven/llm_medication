#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.search.web_search import web_search, search_medical_info
import json


def test_web_search_with_real_api():
    """
    使用真实API密钥测试web搜索功能
    请在运行前设置正确的API密钥
    """
    print("=== Web搜索功能实际测试 ===\n")
    
    # 请替换为您的真实API密钥
    api_key = "sk-458cada7190d42e0aff052c288ab05c5"  # 在这里填入您的真实API密钥
    
    if api_key == "sk-********":
        print("警告: 请先在test_web_search.py文件中设置真实的API密钥!")
        print("将第20行的api_key变量替换为您的实际API密钥")
        return
    
    test_queries = [
        "天空为什么是蓝色的？",
        "Python编程入门",
        "人工智能的发展历史"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: 搜索 '{query}'")
        try:
            result = web_search(
                query=query,
                summary=True,
                count=3,
                api_key=api_key
            )
            
            print("✓ 搜索成功!")
            print(f"返回数据类型: {type(result)}")
            
            if isinstance(result, dict):
                print(f"包含的键: {list(result.keys())}")
                
                # 尝试解析常见的响应格式
                if 'results' in result:
                    print(f"搜索结果数量: {len(result['results'])}")
                elif 'data' in result:
                    print(f"数据条目数量: {len(result['data'])}")
                
                # 显示部分结果内容（避免输出过长）
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                if len(result_str) > 500:
                    print("结果内容 (前500字符):")
                    print(result_str[:500] + "...")
                else:
                    print("完整结果内容:")
                    print(result_str)
            
        except Exception as e:
            print(f"✗ 搜索失败: {str(e)}")
        
        print("-" * 60)


def test_medical_search():
    """
    测试医疗信息搜索功能
    """
    print("=== 医疗信息搜索测试 ===\n")
    
    medical_queries = [
        "高血压的症状",
        "糖尿病的治疗方法",
        "感冒和流感的区别"
    ]
    
    for query in medical_queries:
        print(f"医疗搜索: '{query}'")
        try:
            # 注意: 这里仍然需要有效的API密钥
            result = search_medical_info(query, count=2)
            print("✓ 医疗搜索成功!")
            print(f"返回数据类型: {type(result)}")
            
        except Exception as e:
            print(f"✗ 医疗搜索失败: {str(e)}")
        
        print("-" * 40)


def demo_usage():
    """
    演示如何在其他模块中使用web搜索功能
    """
    print("=== 使用示例演示 ===\n")
    
    # 示例1: 基本搜索
    print("示例1: 基本搜索用法")
    print("""
from src.search.web_search import web_search

# 基本搜索
result = web_search("搜索内容", summary=True, count=5, api_key="your_api_key")

# 或者设置环境变量后直接使用
import os
os.environ['BOCHAAI_API_KEY'] = 'your_api_key'
result = web_search("搜索内容")
""")
    
    # 示例2: 医疗搜索
    print("示例2: 医疗信息搜索用法")
    print("""
from src.search.web_search import search_medical_info

# 医疗信息搜索（会自动添加医疗相关关键词）
result = search_medical_info("疾病症状查询", count=3)
""")
    
    # 示例3: 错误处理
    print("示例3: 错误处理")
    print("""
try:
    result = web_search("查询内容")
    # 处理搜索结果
    if 'results' in result:
        for item in result['results']:
            print(item.get('title', ''))
            print(item.get('content', ''))
except ValueError as e:
    print(f"API密钥错误: {e}")
except requests.RequestException as e:
    print(f"网络请求错误: {e}")
""")


if __name__ == "__main__":
    print("Web搜索模块测试程序\n")
    
    # 演示用法
    demo_usage()
    
    # 提示用户设置API密钥
    print("\n" + "="*60)
    print("要运行实际测试，请:")
    print("1. 获取博茶AI的API密钥")
    print("2. 修改本文件第20行，设置真实的API密钥")
    print("3. 重新运行此脚本")
    print("="*60)
    
    # 尝试运行实际测试（需要真实API密钥）
    test_web_search_with_real_api() 