import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.rewrite_disease_cause import rewrite_disease_cause

def test_rewrite_disease_cause():
    """测试病因简化功能"""
    print("=== 病因简化测试 ===\n")
    
    # 测试用例：高血压病因
    test_cases = [
        {
            "disease": "高血压",
            "raw_cause": "约75%的原发性高血压患者具有遗传素质，同一家族中高血压患者常集中出现。其血清中有一种激素样物质，可抑制Na+/K+-ATP酶活性，以致钠钾泵功能降低，导致细胞内Na+、Ca2+浓度增加，动脉壁SMC收缩加强，肾上腺素能受体密度增加，血管反应性加强。一般而言，日均摄盐量高的人群，其血压升高百分率或平均血压高于摄盐量低者。"
        },
        {
            "disease": "胃炎", 
            "raw_cause": "胃炎的病因复杂，包括幽门螺杆菌感染、非甾体抗炎药物使用、酒精摄入、压力等。幽门螺杆菌通过产生尿素酶、细胞毒素等毒力因子，破坏胃黏膜屏障，引起炎症反应。非甾体抗炎药抑制环氧合酶，减少前列腺素合成，削弱胃黏膜保护机制。"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试案例 {i}: {case['disease']}")
        print("-" * 40)
        print(f"原始病因: {case['raw_cause'][:100]}...")
        
        try:
            simplified = rewrite_disease_cause(
                raw_cause=case['raw_cause'],
                disease_name=case['disease']
            )
            print(f"简化病因: {simplified}")
            print(f"字符数: {len(simplified)}")
            print("✓ 测试成功")
        except Exception as e:
            print(f"✗ 测试失败: {str(e)}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_rewrite_disease_cause()
