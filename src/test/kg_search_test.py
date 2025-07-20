#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.search.kg_search import search_diseases_by_symptoms

def test_symptom_search():
    """
    测试基于症状的疾病搜索功能
    """
    print("=== 知识图谱症状搜索测试 ===\n")
    
    # 测试用例1：多个症状
    test_symptoms_1 = ["胸痛", "呼吸困难"]
    print(f"测试用例1 - 输入症状: {test_symptoms_1}")
    print("-" * 50)
    
    try:
        results = search_diseases_by_symptoms(test_symptoms_1)
        
        if results:
            print(f"找到 {len(results)} 个相关疾病:\n")
            for i, disease in enumerate(results, 1):
                print(f"{i}. 疾病名称: {disease['name']}")
                print(f"   描述: {disease['desc'][:100]}..." if len(disease['desc']) > 100 else f"   描述: {disease['desc']}")
                print(f"   症状: {disease['symptom']}")
                print(f"   病因: {disease['cause'][:50]}..." if len(disease['cause']) > 50 else f"   病因: {disease['cause']}")
                print(f"   治疗科室: {disease['cure_department']}")
                print(f"   并发症: {disease['acompany']}")
                print()
        else:
            print("未找到相关疾病")
            
    except Exception as e:
        print(f"测试失败: {e}")
    
    print("=" * 60)
    
    # 测试用例2：单个症状
    test_symptoms_2 = ["发热"]
    print(f"\n测试用例2 - 输入症状: {test_symptoms_2}")
    print("-" * 50)
    
    try:
        results = search_diseases_by_symptoms(test_symptoms_2)
        
        if results:
            print(f"找到 {len(results)} 个相关疾病:")
            for i, disease in enumerate(results, 1):
                print(f"{i}. {disease['name']} - 症状匹配: {[s for s in disease['symptom'] if s in test_symptoms_2]}")
        else:
            print("未找到相关疾病")
            
    except Exception as e:
        print(f"测试失败: {e}")

def test_interactive():
    """
    交互式测试
    """
    print("\n=== 交互式症状搜索测试 ===")
    print("输入症状，用逗号分隔（如：胸痛,呼吸困难），输入'exit'退出")
    
    while True:
        user_input = input("\n请输入症状: ").strip()
        
        if user_input.lower() == 'exit':
            print("测试结束")
            break
            
        if not user_input:
            print("请输入有效的症状")
            continue
        
        # 解析输入的症状
        symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
        
        print(f"\n搜索症状: {symptoms}")
        print("-" * 40)
        
        try:
            results = search_diseases_by_symptoms(symptoms)
            
            if results:
                print(f"找到 {len(results)} 个相关疾病:\n")
                for i, disease in enumerate(results, 1):
                    print(f"{i}. {disease['name']}")
                    print(f"   匹配症状: {[s for s in disease['symptom'] if s in symptoms]}")
                    print(f"   所有症状: {disease['symptom']}")
                    print()
            else:
                print("未找到相关疾病")
                
        except Exception as e:
            print(f"搜索失败: {e}")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 预设测试用例")
    print("2. 交互式测试")
    
    choice = input("请选择(1/2): ").strip()
    
    if choice == '1':
        test_symptom_search()
    elif choice == '2':
        test_interactive()
    else:
        print("无效选择，运行预设测试用例")
        test_symptom_search()
