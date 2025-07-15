#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.rewrite_query import process_dialog_symptoms

def test_symptom_extraction():
    """测试症状提取功能"""
    
    # 测试案例
    test_cases = [
        {
            "name": "急性胃肠炎",
            "dialog": "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛\n医生：那估计是喝酒引起的急性胃肠炎"
        },
        {
            "name": "肠易激综合症", 
            "dialog": "患者：喝热奶茶疼的很厉害，稀便，肚子痛\n患者：痛的时候是绞痛\n医生：那考虑肠易激综合症"
        }
    ]
    
    print("=== 症状提取测试 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试案例{i} - {case['name']}：")
        try:
            symptoms = process_dialog_symptoms(case['dialog'])
            print(f"提取的症状：{symptoms}")
            print(f"症状数量：{len(symptoms)}\n")
        except Exception as e:
            print(f"测试失败：{e}\n")

if __name__ == "__main__":
    test_symptom_extraction()
