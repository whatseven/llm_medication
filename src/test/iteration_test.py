#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import json

# 设置正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
llm_medication_path = os.path.join(project_root, 'llm_medication')
sys.path.append(llm_medication_path)

from src.model.iteration import iterative_diagnose, extract_diagnostic_suggestions


def test_iterative_diagnose():
    """测试R1专家评估功能"""
    
    print("=== 环境信息检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目路径: {llm_medication_path}")
    print("-" * 50)
    
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
    
    print("=== 开始测试R1专家评估 ===")
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
        
        print("\n=== 专家评估结果分析 ===")
        print(f"返回结果类型: {type(result)}")
        print(f"返回结果内容: {result}")
        
        # 详细分析结果
        if isinstance(result, dict):
            print(f"是否正确: {result.get('is_correct', 'N/A')}")
            
            if 'diagnostic_suggestions' in result:
                suggestions = result['diagnostic_suggestions']
                print(f"诊断建议类型: {type(suggestions)}")
                print(f"诊断建议内容: {suggestions}")
                
                if isinstance(suggestions, dict):
                    print(f"建议疾病: {suggestions.get('recommended_diseases', 'N/A')}")
                    print(f"建议原因: {suggestions.get('reason', 'N/A')}")
            else:
                print("未包含诊断建议信息")
        
        # 验证返回格式
        assert isinstance(result, dict), "返回结果应该是字典"
        assert 'is_correct' in result, "应包含is_correct字段"
        assert isinstance(result['is_correct'], bool), "is_correct应该是布尔值"
        
        print("\n✅ 基本测试通过！函数正常工作")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())


def test_diagnostic_suggestions_parsing():
    """专门测试诊断建议解析功能"""
    
    print("\n" + "="*60)
    print("=== 诊断建议解析测试 ===")
    
    # 测试不同格式的建议内容
    test_cases = [
        # 正确格式
        '''<diagnostic_suggestions>
{"recommended_diseases": ["急性胃肠炎", "食物中毒"], "reason": "症状匹配度高"}
</diagnostic_suggestions>''',
        
        # 常见错误格式1：缺少引号
        '''<diagnostic_suggestions>
{recommended_diseases: ["急性胃肠炎", "食物中毒"], reason: "症状匹配度高"}
</diagnostic_suggestions>''',
        
        # 常见错误格式2：包含中文引号
        '''<diagnostic_suggestions>
{"recommended_diseases": ["急性胃肠炎", "食物中毒"], "reason": "症状匹配度"高""}
</diagnostic_suggestions>''',
        
        # 常见错误格式3：换行格式
        '''<diagnostic_suggestions>
{
    "recommended_diseases": ["急性胃肠炎", "食物中毒"],
    "reason": "症状匹配度高"
}
</diagnostic_suggestions>''',
    ]
    
    for i, test_content in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print("测试内容:")
        print(test_content)
        
        try:
            result = extract_diagnostic_suggestions(test_content)
            print(f"解析结果: {result}")
            print(f"解析状态: ✅ 成功")
        except Exception as e:
            print(f"解析失败: {e}")
            print(f"解析状态: ❌ 失败")
            
            # 尝试手动提取和分析
            pattern = r'<diagnostic_suggestions>\s*(\{.*?\})\s*</diagnostic_suggestions>'
            match = re.search(pattern, test_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                print(f"提取的JSON字符串: '{json_str}'")
                print(f"JSON字符串长度: {len(json_str)}")
                print(f"JSON字符串repr: {repr(json_str)}")


def test_actual_r1_response():
    """测试实际的R1响应解析"""
    
    print("\n" + "="*60)
    print("=== 实际R1响应测试 ===")
    
    # 模拟一个可能导致错误的诊断（故意让R1驳回）
    symptoms = "患者轻微头痛"
    vector_results = """1. 脑肿瘤 - 症状匹配度: 0.95"""
    graph_data = ""
    doctor_diagnosis = "脑肿瘤"  # 明显过度诊断，应该被驳回
    
    print("使用过度诊断案例来触发R1驳回和建议生成...")
    print(f"症状: {symptoms}")
    print(f"诊断: {doctor_diagnosis}")
    
    try:
        # 直接调用iteration函数，但捕获中间过程
        from src.model.iteration import iterative_diagnose
        from src.model.config import MODELS
        from src.model.prompt import R1_EXPERT_EVALUATION_PROMPT
        from openai import OpenAI
        
        # 构建提示词
        prompt = R1_EXPERT_EVALUATION_PROMPT.format(
            symptoms=symptoms,
            vector_results=vector_results,
            graph_data=graph_data,
            doctor_diagnosis=doctor_diagnosis,
            disease_list=""
        )
        
        print("\n发送给R1的提示词:")
        print("-" * 30)
        print(prompt)
        print("-" * 30)
        
        # 调用R1
        model_config = MODELS["deepseek"]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位资深医疗专家，需要进行推理分析诊断是否正确。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            stream=False
        )
        
        # 获取原始响应
        raw_response = response.choices[0].message.content
        print("\nR1原始响应:")
        print("-" * 30)
        print(raw_response)
        print("-" * 30)
        
        # 尝试解析
        result = extract_diagnostic_suggestions(raw_response)
        print(f"\n建议解析结果: {result}")
        
    except Exception as e:
        print(f"实际测试失败: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    # 运行所有测试
    test_iterative_diagnose()
    test_diagnostic_suggestions_parsing()
    test_actual_r1_response()
