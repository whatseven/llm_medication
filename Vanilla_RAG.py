import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.rewrite_query import process_dialog_symptoms
from src.search.milvus_search import search_similar_diseases
from src.model.doctor import diagnose

def vanilla_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    基于纯向量库的医疗诊断流程（Vanilla RAG）
    跳过analyzer、重排序和图数据库查询，直接基于向量搜索进行诊断
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("开始Vanilla RAG医疗诊断流程...")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 症状提取和改写
        print("\n步骤1: 症状提取和改写...")
        symptoms = process_dialog_symptoms(user_input, model_name)
        print(f"提取到的症状: {symptoms}")
        
        # 将症状列表转换为字符串用于向量搜索
        symptoms_text = ' '.join(symptoms) if symptoms else user_input
        
        # 步骤2: 向量搜索
        print("\n步骤2: 向量搜索...")
        milvus_results = search_similar_diseases(symptoms_text, top_k=3)
        print(f"搜索到 {len(milvus_results)} 个疾病")
        
        if not milvus_results:
            return "未找到相关疾病信息，请咨询专业医生。"
        
        # 步骤3: 直接进行最终诊断（跳过重排序、analyzer和图数据库查询）
        print("\n步骤3: 最终诊断（基于向量库结果）...")
        # 传入空的graph_data字典，表示不使用图数据库信息
        diagnosis_result = diagnose(user_input, milvus_results, {}, model_name, disease_list_file)
        
        print("\nVanilla RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"Vanilla RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # 示例调用
    test_input = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    result = vanilla_rag_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
    
    print("\n" + "="*60)
    print("对比测试：复杂病例")
    
    # 复杂病例测试
    complex_test_input = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    result2 = vanilla_rag_pipeline(complex_test_input)
    print(f"\n复杂病例诊断结果:\n{result2}")
