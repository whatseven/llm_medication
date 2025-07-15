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

def medical_diagnosis_pipeline(user_input: str, model_name: str = None) -> str:
    """
    完整的医疗诊断流程
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
    
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
        milvus_results = search_similar_diseases(symptoms_text, top_k=3)
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
        diagnosis_result = diagnose(user_input, filtered_results, graph_data, model_name)
        
        print("\n诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # 示例调用
    test_input = "患者：鼻子喘气很热，鼻孔处特别干燥, 医生：你好，目前症状多长时间了？有没有鼻涕鼻塞之类的？最近有没有感冒？, 患者：吹了风会有点流鼻涕，没有咳嗽发烧的症状, 医生：那可能还是有些鼻炎表现，与季节干燥有关，可以用[MASK]"
    result = medical_diagnosis_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
