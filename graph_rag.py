import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.rewrite_query import process_dialog_symptoms
from src.search.kg_search import search_diseases_by_symptoms
from src.model.rewrite_disease_cause import rewrite_disease_cause
from src.model.doctor import diagnose

def graph_rag_diagnosis(user_input: str, model_name: str = None, disease_list_file: str = None, silent_mode: bool = False) -> str:
    """
    基于图数据库的纯Graph RAG诊断流程
    
    Args:
        user_input: 医患对话内容
        model_name: 使用的模型名称，可选
        disease_list_file: 疾病列表文件路径，可选
        silent_mode: 静默模式，True时减少日志输出
    
    Returns:
        str: 最终诊断结果
    """
    try:
        if not silent_mode:
            print("=== 开始Graph RAG诊断流程 ===")
            print(f"用户输入: {user_input}")
        
        # 步骤1: 症状提取
        if not silent_mode:
            print("\n步骤1: 症状提取...")
        symptoms = process_dialog_symptoms(user_input, model_name)
        if not silent_mode:
            print(f"提取到的症状: {symptoms}")
        
        # 如果没有提取到症状，使用空列表继续流程
        if not symptoms:
            symptoms = []
            if not silent_mode:
                print("未提取到有效症状，将使用空值进行诊断")
        
        # 步骤2: 图数据库搜索
        if not silent_mode:
            print("\n步骤2: 图数据库搜索...")
        kg_results = search_diseases_by_symptoms(symptoms) if symptoms else []
        if not silent_mode:
            print(f"图数据库搜索到 {len(kg_results)} 个相关疾病")
        
        # 如果没有找到相关疾病，继续流程但使用空值
        
        # 步骤3: 病因简化处理
        if not silent_mode:
            print("\n步骤3: 病因简化处理...")
        processed_diseases = []
        graph_data = {}
        
        # 如果有疾病信息，进行处理
        if kg_results:
            for disease in kg_results:
                disease_name = disease['name']
                if not silent_mode:
                    print(f"处理疾病: {disease_name}")
                
                # 简化病因
                if disease['cause']:
                    simplified_cause = rewrite_disease_cause(
                        raw_cause=disease['cause'],
                        disease_name=disease_name,
                        model_name=model_name
                    )
                    if simplified_cause:
                        disease['cause'] = simplified_cause
                        if not silent_mode:
                            print(f"✓ {disease_name} 病因简化完成")
                    else:
                        if not silent_mode:
                            print(f"✗ {disease_name} 病因简化失败，保留原始病因")
                
                # 转换为doctor期望的格式
                formatted_disease = {
                    'name': disease['name'],
                    'desc': disease['desc'] or '',
                    'symptom': str(disease['symptom']) if disease['symptom'] else '[]',
                    'similarity_score': 0.9  # 图数据库匹配度设为固定高值
                }
                processed_diseases.append(formatted_disease)
                
                # 构建graph_data格式
                graph_info = f"疾病名称：{disease_name}\n\n"
                if disease['cause']:
                    graph_info += f"疾病病因：{disease['cause']}\n\n"
                if disease['cure_department']:
                    graph_info += f"治疗科室：{' '.join(disease['cure_department'])}\n\n"
                if disease['acompany']:
                    graph_info += f"并发症：{' '.join(disease['acompany'])}\n\n"
                
                graph_data[disease_name] = graph_info.strip()
        else:
            # 没有疾病信息时的处理
            if not silent_mode:
                print("图数据库未搜索到相关疾病，将使用空值进行诊断")
        
        if not silent_mode:
            print(f"成功处理 {len(processed_diseases)} 个疾病")
        
        # 步骤4: 最终诊断
        if not silent_mode:
            print("\n步骤4: 最终诊断...")
        diagnosis_result = diagnose(
            user_input=user_input,
            vector_results=processed_diseases,
            graph_data=graph_data,
            model_name=model_name,
            disease_list_file=disease_list_file
        )
        
        if not silent_mode:
            print("\nGraph RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"Graph RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # 简单测试
    test_input = "患者出现胸痛、呼吸困难的症状，持续了几天"
    
    print("=== Graph RAG测试 ===")
    result = graph_rag_diagnosis(test_input)
    print(f"\n最终诊断结果:\n{result}")
