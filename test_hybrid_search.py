#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_hybrid_search():
    """
    测试修改后的混合搜索功能
    """
    print("=== 测试混合搜索功能 ===\n")
    
    try:
        # 导入修改后的搜索函数
        from search.milvus_search_copy import search_similar_diseases
        
        # 测试查询
        test_queries = [
            "腹泻症状",
            "头痛头晕",
            "发热咳嗽",
            "腹痛呕吐"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"测试 {i}: 查询 '{query}'")
            print("-" * 50)
            
            try:
                results = search_similar_diseases(query, top_k=10)
                
                if results:
                    print(f"✓ 混合搜索成功，返回 {len(results)} 个结果")
                    
                    for j, result in enumerate(results, 1):
                        print(f"结果 {j}:")
                        print(f"  疾病名称: {result.get('name', 'Unknown')}")
                        print(f"  相似度分数: {result.get('similarity_score', 0):.4f}")
                        print(f"  OID: {result.get('oid', 'Unknown')}")
                        
                        # 显示部分描述
                        desc = result.get('desc', '')
                        if len(desc) > 100:
                            print(f"  描述: {desc[:100]}...")
                        else:
                            print(f"  描述: {desc}")
                        print()
                else:
                    print("✗ 未找到搜索结果")
                    
            except Exception as e:
                print(f"✗ 搜索失败: {str(e)}")
            
            print("=" * 60)
            print()

    except ImportError as e:
        print(f"导入失败: {str(e)}")
        print("请确保milvus_search_copy.py文件存在且路径正确")


def test_compare_searches():
    """
    比较原始搜索和混合搜索的结果差异
    """
    print("=== 比较搜索结果 ===\n")
    
    try:
        # 导入两个搜索函数进行比较
        from search.milvus_search import search_similar_diseases as original_search
        from search.milvus_search_copy import search_similar_diseases as hybrid_search
        
        query = "腹泻腹痛症状"
        print(f"查询: '{query}'\n")
        
        print("原始搜索结果:")
        print("-" * 30)
        try:
            original_results = original_search(query, top_k=10)
            if original_results:
                for i, result in enumerate(original_results, 1):
                    print(f"{i}. {result.get('name', 'Unknown')} (相似度: {result.get('similarity_score', 0):.4f})")
            else:
                print("无结果")
        except Exception as e:
            print(f"原始搜索失败: {str(e)}")
        
        print("\n混合搜索结果:")
        print("-" * 30)
        try:
            hybrid_results = hybrid_search(query, top_k=10)
            if hybrid_results:
                for i, result in enumerate(hybrid_results, 1):
                    print(f"{i}. {result.get('name', 'Unknown')} (相似度: {result.get('similarity_score', 0):.4f})")
            else:
                print("无结果")
        except Exception as e:
            print(f"混合搜索失败: {str(e)}")
            
    except ImportError as e:
        print(f"导入失败: {str(e)}")


if __name__ == "__main__":
    print("混合搜索测试程序\n")
    
    # 测试混合搜索
    test_hybrid_search()
    
    print("\n" + "="*60)
    
    # 比较搜索结果
    test_compare_searches()
    
    print("\n=== 测试完成 ===")
    print("注意: 需要确保medication2 collection已创建并包含数据") 