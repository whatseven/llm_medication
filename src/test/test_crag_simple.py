#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 设置博茶API密钥环境变量
os.environ['BOCHAAI_API_KEY'] = 'sk-458cada7190d42e0aff052c288ab05c5'

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """测试导入是否正常"""
    print("=== 测试模块导入 ===")
    
    try:
        print("1. 测试web搜索模块...")
        from src.search.web_search import web_search
        print("✓ web_search导入成功")
    except Exception as e:
        print(f"✗ web_search导入失败: {e}")
        return False
    
    try:
        print("2. 测试向量搜索模块...")
        from src.search.milvus_search import search_similar_diseases
        print("✓ milvus_search导入成功")
    except Exception as e:
        print(f"✗ milvus_search导入失败: {e}")
        return False
    
    try:
        print("3. 测试doctor模块...")
        from src.model.doctor import diagnose
        print("✓ doctor模块导入成功")
    except Exception as e:
        print(f"✗ doctor模块导入失败: {e}")
        return False
    
    try:
        print("4. 测试CRAG模块...")
        from crag import corrective_rag_pipeline
        print("✓ CRAG模块导入成功")
    except Exception as e:
        print(f"✗ CRAG模块导入失败: {e}")
        return False
    
    return True

def test_web_search_only():
    """仅测试web搜索功能"""
    print("\n=== 测试Web搜索功能 ===")
    
    try:
        from src.search.web_search import web_search
        
        print("测试查询: '腹泻症状'")
        result = web_search("腹泻症状", count=3)
        
        print(f"返回状态码: {result.get('code', 'unknown')}")
        
        if result.get('code') == 200:
            web_pages = result.get('data', {}).get('webPages', {}).get('value', [])
            print(f"搜索到 {len(web_pages)} 个结果")
            
            for i, page in enumerate(web_pages[:2], 1):  # 只显示前2个结果
                print(f"结果{i}: {page.get('name', 'No title')}")
                print(f"摘要: {page.get('snippet', 'No snippet')[:100]}...")
                print()
        
        return True
        
    except Exception as e:
        print(f"Web搜索测试失败: {e}")
        return False

def test_format_function():
    """测试格式化函数"""
    print("=== 测试格式化函数 ===")
    
    try:
        from crag import format_web_search_results
        
        # 模拟web搜索结果
        mock_result = {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "腹泻的症状和治疗",
                            "snippet": "腹泻是常见的消化系统症状，可能伴有腹痛、恶心等",
                            "url": "https://example.com/1"
                        },
                        {
                            "name": "肠胃炎诊断",
                            "snippet": "急性肠胃炎常表现为腹泻、腹痛、发热等症状",
                            "url": "https://example.com/2"
                        }
                    ]
                }
            }
        }
        
        formatted = format_web_search_results(mock_result)
        print(f"格式化结果数量: {len(formatted)}")
        
        for i, item in enumerate(formatted, 1):
            print(f"项目{i}: {item.get('name', 'No name')}")
            print(f"来源: {item.get('source', 'unknown')}")
            print()
        
        return True
        
    except Exception as e:
        print(f"格式化测试失败: {e}")
        return False

if __name__ == "__main__":
    print("Corrective RAG 简化测试 (适配lightrag环境)\n")
    
    # 测试导入
    if not test_imports():
        print("模块导入失败，请检查环境配置")
        sys.exit(1)
    
    # 测试web搜索
    print()
    if not test_web_search_only():
        print("Web搜索功能异常")
    
    # 测试格式化
    print()
    if not test_format_function():
        print("格式化功能异常")
    
    print("\n=== 基础功能测试完成 ===")
    print("如果上述测试都通过，说明CRAG的基础组件工作正常")
    print("可以尝试运行完整的CRAG流程: python crag.py") 