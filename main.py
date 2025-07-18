import sys
import os

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.rewrite_query import process_dialog_symptoms
from src.search.milvus_search import search_similar_diseases
from src.rerank.reranker import rerank_diseases
from src.model.analyzer import analyze_diagnosis
from src.search.neo4j_diagnose import neo4j_diagnosis_search
from src.model.doctor import diagnose

def medical_diagnosis_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    完整的医疗诊断流程
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("开始医疗诊断流程...")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 症状提取和改写
        print("\n步骤1: 症状提取和改写...")
        symptoms = process_dialog_symptoms(user_input, model_name)
        print(f"提取到的症状: {symptoms}")
        
        # 将症状列表转换为字符串用于向量搜索
        symptoms_text = ' '.join(symptoms) if symptoms else user_input
        
        # 步骤2: 向量搜索
        print("\n步骤2: 向量搜索...")
        milvus_results = search_similar_diseases(symptoms_text, top_k=5)
        print(f"搜索到 {len(milvus_results)} 个疾病")
        
        if not milvus_results:
            return "未找到相关疾病信息，请咨询专业医生。"
        
        # 步骤3: 重排序
        print("\n步骤3: 重排序...")
        reranked_results = rerank_diseases(symptoms_text, milvus_results)
        print(f"重排序完成，共 {len(reranked_results)} 个结果")
        
        # 步骤4: 分析是否需要更多信息
        print("\n步骤4: 分析诊断...")
        analysis_result = analyze_diagnosis(user_input, reranked_results, model_name)
        print(f"分析结果: {analysis_result}")
        
        if 'error' in analysis_result:
            raise Exception(analysis_result['error'])
        
        need_more_info = analysis_result.get('need_more_info', False)
        target_diseases = analysis_result.get('diseases', [])
        
        # 根据分析结果决定流程
        if need_more_info and target_diseases:
            print(f"\n需要更多信息，目标疾病: {target_diseases}")
            
            # 步骤5: 图数据库查询
            print("\n步骤5: 图数据库查询...")
            graph_data = {}
            for disease_name in target_diseases:
                print(f"查询疾病: {disease_name}")
                disease_info = neo4j_diagnosis_search(disease_name)
                if disease_info:
                    graph_data[disease_name] = disease_info
            
            # 过滤向量库结果，只保留目标疾病
            filtered_results = []
            for result in reranked_results:
                if result.get('name') in target_diseases:
                    filtered_results.append(result)
            
            print(f"过滤后的向量库结果: {len(filtered_results)} 个")
            
        else:
            print("\n无需更多信息，直接诊断")
            # 不需要更多信息，使用所有reranked结果
            filtered_results = reranked_results
            graph_data = {}
        
        # 步骤6: 最终诊断
        print("\n步骤6: 最终诊断...")
        diagnosis_result = diagnose(user_input, filtered_results, graph_data, model_name, disease_list_file)
        
        print("\n诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # 示例调用
    test_input = "患者病历：\n患者于入院前3月，出现因食辛辣醇厚且劳累后出现肛旁肿胀疼痛，症情反复发作渐加重，遂来本院求治。刻下：肛旁肿胀疼痛剧烈，坐卧不宁，行走不利。大便，日行1次，质软，排出畅，伴便血，量少，色鲜红，无排便不尽及肛门坠胀感，无粘液便，小溲畅，无发热恶寒。纳食可，夜寐尚可，舌红，苔黄，脉滑数。\n患者主诉：\n肛旁肿痛3月。\n患者四诊信息：\n神志清晰，精神尚可，形体形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌红，苔黄，脉滑数。"
    result = medical_diagnosis_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
