import sys
import os

# 添加src路径到系统路径,没有设置相似度
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.rewrite_query import process_dialog_symptoms
from src.search.kg_search import search_diseases_by_symptoms
from src.model.doctor import diagnose
from src.rerank.reranker import rerank_diseases
from src.embedding.embedding import get_embedding

def graph_rag_diagnosis(user_input: str, model_name: str = None, disease_list_file: str = None, silent_mode: bool = False) -> str:
    """
    基于图数据库的Graph RAG诊断流程，包含语义重排序
    
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
            print("=== 开始Graph RAG诊断流程（含语义重排序）===")
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
        
        # 步骤3: 语义重排序
        if not silent_mode:
            print("\n步骤3: 语义重排序...")
        processed_diseases = []
        graph_data = {}
        
        # 如果有疾病信息，进行处理
        if kg_results:
            # 将图数据库结果转换为reranker期望的格式
            milvus_format_results = []
            for disease in kg_results:
                milvus_format_item = {
                    'name': disease['name'],
                    'desc': disease['desc'] or '',
                    'symptom': str(disease['symptom']) if disease['symptom'] else '[]'
                }
                milvus_format_results.append(milvus_format_item)
            
            # 使用user_input进行重排序
            try:
                if not silent_mode:
                    print("正在进行语义重排序...")
                reranked_results = rerank_diseases(user_input, milvus_format_results)
                if not silent_mode:
                    print(f"重排序完成，获得 {len(reranked_results)} 个结果")
            except Exception as e:
                if not silent_mode:
                    print(f"重排序失败: {e}，使用默认相似度")
                # 重排序失败时，添加默认相似度
                reranked_results = []
                for item in milvus_format_results:
                    item_with_score = item.copy()
                    item_with_score['similarity_score'] = 0.6  # 默认相似度
                    reranked_results.append(item_with_score)
            
            # 处理重排序结果，构建最终数据
            for i, reranked_item in enumerate(reranked_results):
                # 找到对应的原始图数据库结果
                original_disease = None
                for disease in kg_results:
                    if disease['name'] == reranked_item['name']:
                        original_disease = disease
                        break
                
                if original_disease is None:
                    continue
                
                disease_name = original_disease['name']
                if not silent_mode:
                    similarity_score = reranked_item.get('relevance_score', reranked_item.get('similarity_score', 0.6))
                    print(f"处理疾病: {disease_name} (相似度: {similarity_score:.4f})")
                
                # 转换为doctor期望的格式，使用重排序得到的相似度
                formatted_disease = {
                    'name': original_disease['name'],
                    'desc': original_disease['desc'] or '',
                    'symptom': str(original_disease['symptom']) if original_disease['symptom'] else '[]',
                    'similarity_score': reranked_item.get('relevance_score', reranked_item.get('similarity_score', 0.6))
                }
                processed_diseases.append(formatted_disease)
                
                # 构建graph_data格式（使用原始病因，不进行缩写）
                graph_info = f"疾病名称：{disease_name}\n\n"
                if original_disease['desc']:
                    graph_info += f"疾病描述：{original_disease['desc']}\n\n"
                if original_disease['symptom']:
                    graph_info += f"相关症状：{original_disease['symptom']}\n\n"
                if original_disease['cause']:
                    graph_info += f"疾病病因：{original_disease['cause']}\n\n"
                if original_disease['cure_department']:
                    graph_info += f"治疗科室：{' '.join(original_disease['cure_department'])}\n\n"
                if original_disease['acompany']:
                    graph_info += f"并发症：{' '.join(original_disease['acompany'])}\n\n"
                
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
