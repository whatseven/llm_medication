import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from main import medical_diagnosis_pipeline

def test_diagnosis():
    """
    测试医疗诊断流程
    """
    print("=== 医疗诊断系统测试 ===")
    print("请输入您的症状描述，输入'exit'退出测试")
    
    while True:
        # 获取用户输入
        user_input = input("\n请输入症状描述: ").strip()
        
        if user_input.lower() == 'exit':
            print("测试结束")
            break
            
        if not user_input:
            print("请输入有效的症状描述")
            continue
        
        print(f"\n{'='*60}")
        print("开始诊断流程...")
        print(f"{'='*60}")
        
        try:
            # 调用诊断流程
            result = medical_diagnosis_pipeline(user_input)
            
            print(f"\n{'='*60}")
            print("诊断结果:")
            print(f"{'='*60}")
            print(result)
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"测试过程中出现错误: {str(e)}")

def test_with_custom_input():
    """
    单次测试函数
    """
    # 可以在这里直接修改测试用例
    test_cases = [
        "我最近腹泻很严重，伴有腹痛，肛门也很疼",
        "我感觉胸闷气短，呼吸困难",
        "腹部胀痛，恶心想吐"
    ]
    
    print("=== 预设测试用例 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case}")
    
    choice = input("\n选择测试用例编号(1-3)，或输入'c'自定义输入: ").strip()
    
    if choice == 'c':
        user_input = input("请输入自定义症状描述: ").strip()
    elif choice in ['1', '2', '3']:
        user_input = test_cases[int(choice) - 1]
    else:
        print("无效选择")
        return
    
    print(f"\n{'='*60}")
    print(f"测试输入: {user_input}")
    print(f"{'='*60}")
    
    try:
        result = medical_diagnosis_pipeline(user_input)
        print(f"\n{'='*60}")
        print("诊断结果:")
        print(f"{'='*60}")
        print(result)
        print(f"{'='*60}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 交互式测试(可多次输入)")
    print("2. 单次测试(预设用例或自定义)")
    
    mode = input("请选择模式(1/2): ").strip()
    
    if mode == '1':
        test_diagnosis()
    elif mode == '2':
        test_with_custom_input()
    else:
        print("无效选择，默认使用交互式测试")
        test_diagnosis()
