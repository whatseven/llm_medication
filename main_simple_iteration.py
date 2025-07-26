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
from src.model.rewrite_disease_cause import rewrite_disease_cause
from src.model.iteration import iterative_diagnose

def parse_neo4j_result(neo4j_text: str) -> dict:
    """
    解析图数据库返回的格式化文本
    
    Args:
        neo4j_text: 图数据库返回的格式化文本
    
    Returns:
        dict: 解析后的结构化信息
    """
    result = {
        'disease_name': '',
        'cause': '',
        'department': '',
        'complications': ''
    }
    
    try:
        lines = neo4j_text.strip().split('\n')
        current_field = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('疾病名称：'):
                result['disease_name'] = line.replace('疾病名称：', '').strip()
            elif line.startswith('疾病病因：'):
                current_field = 'cause'
                result['cause'] = line.replace('疾病病因：', '').strip()
            elif line.startswith('治疗科室：'):
                current_field = 'department'
                result['department'] = line.replace('治疗科室：', '').strip()
            elif line.startswith('并发症：'):
                current_field = 'complications'
                result['complications'] = line.replace('并发症：', '').strip()
            elif current_field and line:
                # 续行内容
                result[current_field] += line
                
    except Exception as e:
        print(f"解析图数据库结果出错: {str(e)}")
        
    return result

def process_graph_data_with_simplified_cause(disease_name: str, neo4j_text: str, model_name: str = None) -> str:
    """
    处理图数据库信息，简化病因并重新组装
    
    Args:
        disease_name: 疾病名称
        neo4j_text: 图数据库原始返回文本
        model_name: 用于病因简化的模型名称
    
    Returns:
        str: 处理后的图数据库信息，如果处理失败返回空字符串
    """
    try:
        # 解析原始信息
        parsed_data = parse_neo4j_result(neo4j_text)
        
        if not parsed_data['cause']:
            print(f"警告: 疾病 {disease_name} 没有病因信息")
            return ""
        
        # 简化病因
        simplified_cause = rewrite_disease_cause(
            raw_cause=parsed_data['cause'],
            disease_name=disease_name,
            model_name=model_name
        )
        
        if not simplified_cause:
            print(f"警告: 疾病 {disease_name} 病因简化失败，跳过该疾病")
            return ""
        
        # 重新组装信息
        result_text = f"疾病名称：{disease_name}\n\n"
        result_text += f"疾病病因：{simplified_cause}\n\n"
        
        if parsed_data['department']:
            result_text += f"治疗科室：{parsed_data['department']}\n\n"
        
        if parsed_data['complications']:
            result_text += f"并发症：{parsed_data['complications']}\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        print(f"处理疾病 {disease_name} 的图数据库信息出错: {str(e)}，跳过该疾病")
        return ""

def get_initial_diagnosis_data(user_input: str, model_name: str = None, top_k: int = 5, silent_mode: bool = False) -> dict:
    """
    获取初始诊断所需的所有数据（只执行一次）
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        top_k (int): 向量搜索返回的数量
        silent_mode (bool): 静默模式，减少日志输出
    
    Returns:
        dict: 包含所有诊断数据的字典
    """
    try:
        if not silent_mode:
            print("获取初始诊断数据...")
            print(f"用户输入: {user_input}")
        
        # 步骤1: 症状提取和改写
        if not silent_mode:
            print("\n步骤1: 症状提取和改写...")
        symptoms = process_dialog_symptoms(user_input, model_name)
        if not silent_mode:
            print(f"提取到的症状: {symptoms}")
        
        # 将症状列表转换为字符串用于向量搜索
        symptoms_text = ' '.join(symptoms) if symptoms else user_input
        
        # 步骤2: 向量搜索
        if not silent_mode:
            print(f"\n步骤2: 向量搜索(top_k={top_k})...")
        milvus_results = search_similar_diseases(symptoms_text, top_k=top_k)
        if not silent_mode:
            print(f"搜索到 {len(milvus_results)} 个疾病")
        
        if not milvus_results:
            return {
                "symptoms": symptoms,
                "vector_results": [],
                "graph_data": {},
                "success": False,
                "error": "未找到相关疾病信息，请咨询专业医生。"
            }
        
        # 步骤3: 重排序
        if not silent_mode:
            print("\n步骤3: 重排序...")
        reranked_results = rerank_diseases(symptoms_text, milvus_results)
        if not silent_mode:
            print(f"重排序完成，共 {len(reranked_results)} 个结果")
        
        # 步骤4: 分析是否需要更多信息
        if not silent_mode:
            print("\n步骤4: 分析诊断...")
        analysis_result = analyze_diagnosis(user_input, reranked_results, model_name)
        if not silent_mode:
            print(f"分析结果: {analysis_result}")
        
        if 'error' in analysis_result:
            return {
                "symptoms": symptoms,
                "vector_results": reranked_results,
                "graph_data": {},
                "success": False,
                "error": analysis_result['error']
            }
        
        need_more_info = analysis_result.get('need_more_info', False)
        target_diseases = analysis_result.get('diseases', [])
        
        # 根据分析结果决定是否查询图数据库
        if need_more_info and target_diseases:
            if not silent_mode:
                print(f"\n需要更多信息，目标疾病: {target_diseases}")
            
            # 步骤5: 图数据库查询和病因简化
            if not silent_mode:
                print("\n步骤5: 图数据库查询和病因简化...")
            graph_data = {}
            for disease_name in target_diseases:
                if not silent_mode:
                    print(f"查询疾病: {disease_name}")
                disease_info = neo4j_diagnosis_search(disease_name)
                if disease_info:
                    # 处理图数据库信息，简化病因
                    processed_info = process_graph_data_with_simplified_cause(
                        disease_name, disease_info, model_name
                    )
                    if processed_info:
                        graph_data[disease_name] = processed_info
                        if not silent_mode:
                            print(f"✓ 疾病 {disease_name} 信息处理完成")
                    else:
                        if not silent_mode:
                            print(f"✗ 疾病 {disease_name} 信息处理失败，跳过")
                else:
                    if not silent_mode:
                        print(f"✗ 疾病 {disease_name} 未找到图数据库信息")
            
            # 过滤向量库结果，只保留成功处理的目标疾病
            successfully_processed_diseases = list(graph_data.keys())
            filtered_results = []
            for result in reranked_results:
                if result.get('name') in successfully_processed_diseases:
                    filtered_results.append(result)
            
            if not silent_mode:
                print(f"过滤后的向量库结果: {len(filtered_results)} 个")
            
        else:
            if not silent_mode:
                print("\n无需更多信息，直接使用重排序结果")
            # 不需要更多信息，使用所有reranked结果
            filtered_results = reranked_results
            graph_data = {}
        
        if not silent_mode:
            print("\n初始数据获取完成!")
        
        return {
            "symptoms": symptoms,
            "vector_results": filtered_results,
            "graph_data": graph_data,
            "success": True
        }
        
    except Exception as e:
        error_msg = f"获取初始诊断数据出错: {str(e)}"
        if not silent_mode:
            print(error_msg)
        return {
            "symptoms": [],
            "vector_results": [],
            "graph_data": {},
            "success": False,
            "error": error_msg
        }

def medical_diagnosis_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None, silent_mode: bool = False) -> str:
    """
    简化迭代的医疗诊断流程：只获取一次数据，后续迭代仅重新调用doctor诊断
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
        silent_mode (bool): 静默模式，True时减少日志输出，用于批量评估
    
    Returns:
        str: 最终诊断结果
    """
    max_retries = 3
    rejection_count = 0  # 记录被驳回次数
    previous_suggestions = None  # 保存上一轮的诊断建议
    
    if not silent_mode:
        print("=== 开始简化迭代的医疗诊断流程 ===")
    
    # 第一步：获取所有诊断数据（只执行一次）
    if not silent_mode:
        print(f"\n{'='*60}")
        print("获取诊断所需的基础数据...")
        print(f"{'='*60}")
    
    initial_data = get_initial_diagnosis_data(
        user_input=user_input,
        model_name=model_name,
        top_k=5,  # 使用固定的top_k
        silent_mode=silent_mode
    )
    
    if not initial_data["success"]:
        return initial_data.get("error", "获取诊断数据失败")
    
    # 准备R1评估需要的格式化字符串（只准备一次）
    symptoms_str = ' '.join(initial_data["symptoms"]) if initial_data["symptoms"] else user_input
    vector_results_str = ""
    for i, disease in enumerate(initial_data["vector_results"], 1):
        vector_results_str += f"{i}. {disease.get('name', 'Unknown')}\n"
        vector_results_str += f"   描述：{disease.get('desc', 'No description')}\n"
        vector_results_str += f"   症状：{disease.get('symptom', 'No symptoms')}\n"
        vector_results_str += f"   相似度：{disease.get('similarity_score', 0):.3f}\n\n"
    
    graph_data_str = ""
    for disease_name, disease_info in initial_data["graph_data"].items():
        graph_data_str += f"{disease_info}\n\n"
    
    if not silent_mode:
        print("基础数据获取完成，开始迭代诊断...")
    
    # 第二步：迭代诊断（使用相同的数据，只重新调用doctor模块）
    for attempt in range(max_retries):
        if not silent_mode:
            print(f"\n{'='*60}")
            print(f"第 {attempt + 1} 次诊断尝试")
            if previous_suggestions and not silent_mode:
                print(f"使用上轮建议：{previous_suggestions.get('recommended_diseases', [])}")
            print(f"{'='*60}")
        
        try:
            # 调用doctor模块进行诊断（使用相同的数据）
            if not silent_mode:
                print("调用doctor模块进行诊断...")
            
            diagnosis_result = diagnose(
                user_input, 
                initial_data["vector_results"], 
                initial_data["graph_data"], 
                model_name, 
                disease_list_file, 
                previous_suggestions  # 传递上轮建议
            )
            
            if not silent_mode:
                print(f"诊断完成: {diagnosis_result[:100]}...")
            
            # R1专家评估
            if not silent_mode:
                print(f"\n{'='*40}")
                print("R1专家评估诊断质量...")
                print(f"{'='*40}")
            
            expert_review = iterative_diagnose(
                symptoms=symptoms_str,
                vector_results=vector_results_str,
                graph_data=graph_data_str,
                doctor_diagnosis=diagnosis_result,
                disease_list_file=disease_list_file
            )
            
            if not silent_mode:
                print(f"R1专家评估结果: {'通过' if expert_review['is_correct'] else '驳回'}")
            
            if expert_review["is_correct"]:
                if not silent_mode:
                    print(f"\n{'='*60}")
                    print("R1专家确认诊断正确，流程结束")
                    print(f"{'='*60}")
                return diagnosis_result
            else:
                rejection_count += 1
                # 提取诊断建议用于下轮重试
                previous_suggestions = expert_review.get("diagnostic_suggestions")
                if not silent_mode and previous_suggestions:
                    print(f"R1专家建议：{previous_suggestions.get('recommended_diseases', [])}")
                
                if attempt == max_retries - 1:  # 这是最后一次尝试
                    if not silent_mode:
                        print("R1专家认为诊断有误，已达到最大重试次数，将启用R1直接诊断...")
                    break  # 跳出循环，进入R1直接诊断
                else:
                    if not silent_mode:
                        print(f"R1专家认为诊断有误，准备第 {attempt + 2} 次重试...")
                
        except Exception as e:
            if not silent_mode:
                print(f"第 {attempt + 1} 次诊断过程出错: {str(e)}")
            continue
    
    # 3次重试都失败或被驳回，让R1直接诊断
    if not silent_mode:
        print(f"\n{'='*60}")
        print(f"知识库诊断方式已无法满足要求 (共被驳回{rejection_count}次)，启用R1直接诊断模式")
        print(f"{'='*60}")
    
    try:
        from src.model.config import MODELS
        from openai import OpenAI
        
        # 直接用普通模型进行诊断
        model_config = MODELS["deepseek"]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 读取疾病列表（如果提供）
        disease_list_str = ""
        if disease_list_file and os.path.exists(disease_list_file):
            with open(disease_list_file, 'r', encoding='utf-8') as f:
                diseases = [line.strip() for line in f.readlines() if line.strip()]
                disease_list_str = f"可选疾病列表：{', '.join(diseases)}"
        
        direct_prompt = f"""你是一位顶级的医疗诊断专家。知识库检索多次失败，现在需要你直接基于医学知识进行诊断。

患者症状：{user_input}
{disease_list_str}

请基于你的医学知识，仔细分析患者症状，给出最可能的诊断：
- 仔细分析症状组合
- 考虑疾病的常见性和症状匹配度
- 如果提供了疾病列表，必须从中选择

请将最终诊断放在<final_diagnosis>标签中：
<final_diagnosis>
{{"diseases": ["疾病名称"]}}
</final_diagnosis>"""

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位资深医疗专家，需要进行深度推理分析。"},
                {"role": "user", "content": direct_prompt}
            ],
            temperature=0.1,
            stream=False
        )
        
        content = response.choices[0].message.content
        if not silent_mode:
            print("R1直接诊断完成")
        return content
        
    except Exception as e:
        return f"R1直接诊断也失败了: {str(e)}"

if __name__ == "__main__":
    # 示例调用
    test_input = "患者病历：\n患者于入院前3月，出现因食辛辣醇厚且劳累后出现肛旁肿胀疼痛，症情反复发作渐加重，遂来本院求治。刻下：肛旁肿胀疼痛剧烈，坐卧不宁，行走不利。大便，日行1次，质软，排出畅，伴便血，量少，色鲜红，无排便不尽及肛门坠胀感，无粘液便，小溲畅，无发热恶寒。纳食可，夜寐尚可，舌红，苔黄，脉滑数。\n患者主诉：\n肛旁肿痛3月。\n患者四诊信息：\n神志清晰，精神尚可，形体形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌红，苔黄，脉滑数。"
    result = medical_diagnosis_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
