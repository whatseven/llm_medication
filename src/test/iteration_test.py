#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 设置正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
llm_medication_path = os.path.join(project_root, 'llm_medication')
sys.path.append(llm_medication_path)

from src.model.iteration import iterative_diagnose


def test_iterative_diagnose():
    """测试R1专家评估功能"""
    
    # 模拟测试数据
    symptoms = "患者出现腹痛、恶心、呕吐的症状"
    
    vector_results = """
    候选疾病：
    1. 急性胃炎 - 症状匹配度: 0.92
    2. 胃溃疡 - 症状匹配度: 0.85
    3. 胆囊炎 - 症状匹配度: 0.78
    """
    
    graph_data = """
    急性胃炎病因：常见于饮食不当、药物刺激等
    典型症状：腹痛、恶心、呕吐、食欲不振
    """
    
    doctor_diagnosis = "急性胃炎"
    
    print("开始测试R1专家评估...")
    print(f"症状: {symptoms}")
    print(f"初步诊断: {doctor_diagnosis}")
    print("-" * 50)
    
    try:
        # 调用专家评估
        print("正在调用DeepSeek R1进行专家评估...")
        import time
        start_time = time.time()
        result = iterative_diagnose(symptoms, vector_results, graph_data, doctor_diagnosis)
        end_time = time.time()
        print(f"R1推理耗时: {end_time - start_time:.2f}秒")
        
        print("专家评估结果:")
        print(f"是否正确: {result['is_correct']}")
        
        # 验证返回格式
        assert isinstance(result, dict), "返回结果应该是字典"
        assert 'is_correct' in result, "应包含is_correct字段"
        assert isinstance(result['is_correct'], bool), "is_correct应该是布尔值"
        
        print("\n✅ 测试通过！函数正常工作")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    test_iterative_diagnose()
