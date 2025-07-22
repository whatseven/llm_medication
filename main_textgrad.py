import sys
import os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加src路径到系统路径  
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
# 同时添加当前目录到路径，确保模块可以被找到
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.model.rewrite_query import process_dialog_symptoms
from src.search.milvus_search import search_similar_diseases
from src.rerank.reranker import rerank_diseases
from src.model.analyzer import analyze_diagnosis
from src.search.neo4j_diagnose import neo4j_diagnosis_search
from src.model.doctor import diagnose
from src.model.rewrite_disease_cause import rewrite_disease_cause
from src.textgrad import LLMMedTextGrad

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

def enhance_disease_with_graph_data(disease_info: dict, disease_name: str, model_name: str = None) -> dict:
    """
    为疾病信息增强图数据库数据
    
    Args:
        disease_info: 基础疾病信息
        disease_name: 疾病名称
        model_name: 模型名称
        
    Returns:
        dict: 增强后的疾病信息
    """
    enhanced_info = disease_info.copy()
    
    try:
        # 查询图数据库
        neo4j_result = neo4j_diagnosis_search(disease_name)
        if neo4j_result:
            # 解析图数据库结果
            parsed_data = parse_neo4j_result(neo4j_result)
            
            # 简化病因
            if parsed_data['cause']:
                simplified_cause = rewrite_disease_cause(
                    raw_cause=parsed_data['cause'],
                    disease_name=disease_name,
                    model_name=model_name
                )
                if simplified_cause:
                    enhanced_info['graph_cause'] = simplified_cause
            
            # 添加其他图数据库信息
            if parsed_data['department']:
                enhanced_info['department'] = parsed_data['department']
            if parsed_data['complications']:
                enhanced_info['complications'] = parsed_data['complications']
                
    except Exception as e:
        print(f"增强疾病 {disease_name} 信息失败: {str(e)}")
    
    return enhanced_info

def medical_diagnosis_pipeline_single(user_input: str, model_name: str = None, disease_list_file: str = None, top_k: int = 5, silent_mode: bool = False, diagnostic_suggestions: dict = None) -> dict:
    """
    单次医疗诊断流程 - TextGrad版本
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
        top_k (int): 向量搜索返回的数量
        silent_mode (bool): 静默模式，减少日志输出
        diagnostic_suggestions (dict): 诊断建议，可选
    
    Returns:
        dict: 包含诊断结果和融合疾病信息的字典
    """
    try:
        if not silent_mode:
            print("开始TextGrad医疗诊断流程...")
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
                "diagnosis": "未找到相关疾病信息，请咨询专业医生。",
                "symptoms": symptoms,
                "disease_information": [],
                "success": False
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
            raise Exception(analysis_result['error'])
        
        need_more_info = analysis_result.get('need_more_info', False)
        target_diseases = analysis_result.get('diseases', [])
        
        # 步骤5: 批量图数据库查询和信息融合（TextGrad版本直接处理所有疾病）
        if not silent_mode:
            print("\n步骤5: 批量图数据库查询和信息融合...")
        
        enhanced_disease_info = []
        diseases_to_process = target_diseases if (need_more_info and target_diseases) else [disease.get('name', '') for disease in reranked_results]
        
        for i, disease_name in enumerate(diseases_to_process):
            if not disease_name:
                continue
                
            if not silent_mode:
                print(f"  处理疾病 {i+1}/{len(diseases_to_process)}: {disease_name}")
            
            # 找到对应的向量库结果
            disease_info = None
            for result in reranked_results:
                if result.get('name') == disease_name:
                    disease_info = result
                    break
            
            if not disease_info:
                if not silent_mode:
                    print(f"  ✗ 疾病 {disease_name} 在向量库结果中未找到")
                continue
            
            # 增强疾病信息
            enhanced_info = enhance_disease_with_graph_data(disease_info, disease_name, model_name)
            enhanced_disease_info.append(enhanced_info)
            
            if 'graph_cause' in enhanced_info:
                if not silent_mode:
                    print(f"  ✓ 疾病 {disease_name} 信息融合完成")
            else:
                if not silent_mode:
                    print(f"  ⚠ 疾病 {disease_name} 仅包含基础信息")
        
        # 步骤6: 获取初步诊断
        if not silent_mode:
            print(f"\n步骤6: 获取初步诊断...")
            print(f"  传入诊断模块的疾病数量: {len(enhanced_disease_info)}")
        
        # 为了兼容现有doctor模块，需要将融合信息重新分离
        vector_results = []
        graph_data = {}
        
        for disease_info in enhanced_disease_info:
            # 构建向量库格式的信息
            vector_item = {
                'name': disease_info.get('name', ''),
                'desc': disease_info.get('desc', ''),
                'symptom': disease_info.get('symptom', ''),
                'similarity_score': disease_info.get('similarity_score', 0)
            }
            vector_results.append(vector_item)
            
            # 如果有图数据库信息，构建图数据库格式
            disease_name = disease_info.get('name', '')
            if disease_name:
                graph_parts = []
                
                if 'graph_cause' in disease_info:
                    graph_parts.append(f"疾病病因：{disease_info['graph_cause']}")
                
                if 'department' in disease_info:
                    graph_parts.append(f"治疗科室：{disease_info['department']}")
                
                if 'complications' in disease_info:
                    graph_parts.append(f"并发症：{disease_info['complications']}")
                
                if graph_parts:
                    graph_data[disease_name] = "\n\n".join(graph_parts)
        
        diagnosis_result = diagnose(user_input, vector_results, graph_data, model_name, disease_list_file, diagnostic_suggestions)
        
        if not silent_mode:
            print("\n单次诊断完成!")
        return {
            "diagnosis": diagnosis_result,
            "symptoms": symptoms,
            "disease_information": enhanced_disease_info,
            "success": True
        }
        
    except Exception as e:
        error_msg = f"诊断流程出错: {str(e)}"
        if not silent_mode:
            print(error_msg)
        return {
            "diagnosis": error_msg,
            "symptoms": [],
            "disease_information": [],
            "success": False
        }

def ensure_diagnosis_format(diagnosis_text: str) -> str:
    """
    确保诊断结果符合标准格式
    
    Args:
        diagnosis_text: 原始诊断文本
        
    Returns:
        格式化的诊断字符串
    """
    if not diagnosis_text or not diagnosis_text.strip():
        return '<final_diagnosis>{"diseases": ["未知疾病"]}</final_diagnosis>'
    
    # 如果已经有正确格式，直接返回
    if '<final_diagnosis>' in diagnosis_text and '</final_diagnosis>' in diagnosis_text:
        return diagnosis_text
    
    # 如果包含错误信息，包装成标准格式
    if "出错" in diagnosis_text or "失败" in diagnosis_text or "错误" in diagnosis_text:
        return f'<final_diagnosis>{{"diseases": ["诊断失败"]}}</final_diagnosis>'
    
    # 清理和提取疾病名称
    clean_text = diagnosis_text.strip()
    
    # 尝试从各种格式中提取疾病名称
    import re
    import json
    
    # 先尝试提取现有的final_diagnosis格式
    pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
    match = re.search(pattern, clean_text, re.DOTALL)
    if match:
        try:
            diagnosis_data = json.loads(match.group(1))
            diseases = diagnosis_data.get('diseases', [])
            if diseases:
                if isinstance(diseases, list):
                    return f'<final_diagnosis>{{"diseases": {json.dumps(diseases, ensure_ascii=False)}}}</final_diagnosis>'
                else:
                    return f'<final_diagnosis>{{"diseases": ["{diseases}"]}}</final_diagnosis>'
        except:
            pass
    
    # 清理常见前缀
    prefixes_to_remove = [
        "优化后的诊断：", "诊断：", "疾病：", "最终诊断：", 
        "建议诊断：", "推荐诊断：", "诊断结果：", "答案："
    ]
    
    for prefix in prefixes_to_remove:
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):].strip()
    
    # 移除引号和标点
    clean_text = clean_text.strip('"\'""''()[]{}（）【】')
    
    # 如果包含多个疾病，取第一个
    if '、' in clean_text:
        clean_text = clean_text.split('、')[0]
    elif '，' in clean_text:
        clean_text = clean_text.split('，')[0]
    elif ',' in clean_text:
        clean_text = clean_text.split(',')[0]
    
    clean_text = clean_text.strip()
    
    if not clean_text:
        clean_text = "未知疾病"
    
    # 包装成标准格式
    return f'<final_diagnosis>{{"diseases": ["{clean_text}"]}}</final_diagnosis>'

def medical_diagnosis_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None, silent_mode: bool = False) -> str:
    """
    带有TextGrad优化的完整医疗诊断流程
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
        silent_mode (bool): 静默模式，True时减少日志输出，用于批量评估
    
    Returns:
        str: 最终诊断结果
    """
    if not silent_mode:
        print("=== 开始TextGrad医疗诊断流程 ===")
    
    try:
        # 步骤1: 获取基础诊断
        if not silent_mode:
            print("\n🔍 执行基础诊断流程...")
        
        base_result = medical_diagnosis_pipeline_single(
            user_input=user_input,
            model_name=model_name,
            disease_list_file=disease_list_file,
            top_k=5,
            silent_mode=silent_mode,
            diagnostic_suggestions=None
        )
        
        if not base_result["success"]:
            if not silent_mode:
                print("❌ 基础诊断失败")
            return ensure_diagnosis_format(base_result["diagnosis"])
        
        if not silent_mode:
            print(f"✅ 获得基础诊断: {base_result['diagnosis']}")
        
        # 步骤2: TextGrad诊断优化
        if not silent_mode:
            print(f"\n🚀 启动TextGrad诊断优化...")
        
        optimizer = LLMMedTextGrad(
            model_name="deepseek",
            num_iterations=1,  # 默认1次迭代
            verbose=not silent_mode
        )
        
        textgrad_result = optimizer.optimize_diagnosis(
            user_input=user_input,
            disease_information=base_result["disease_information"],
            initial_diagnosis=base_result["diagnosis"],
            disease_list_file=disease_list_file
        )
        
        # 步骤3: 返回最终结果
        if textgrad_result["is_correct"]:
            if not silent_mode:
                print(f"✅ TextGrad确认原诊断质量良好")
            return ensure_diagnosis_format(base_result["diagnosis"])
        else:
            optimized_diagnosis = textgrad_result.get("optimized_diagnosis", base_result["diagnosis"])
            if not silent_mode:
                print(f"🎯 TextGrad优化完成: {optimized_diagnosis}")
            
            return ensure_diagnosis_format(optimized_diagnosis)
            
    except Exception as e:
        error_msg = f"TextGrad诊断流程出错: {str(e)}"
        if not silent_mode:
            print(f"❌ {error_msg}")
        return ensure_diagnosis_format(error_msg)

if __name__ == "__main__":
    # 示例调用
    test_input = "患者病历：\n患者于入院前3月，出现因食辛辣醇厚且劳累后出现肛旁肿胀疼痛，症情反复发作渐加重，遂来本院求治。刻下：肛旁肿胀疼痛剧烈，坐卧不宁，行走不利。大便，日行1次，质软，排出畅，伴便血，量少，色鲜红，无排便不尽及肛门坠胀感，无粘液便，小溲畅，无发热恶寒。纳食可，夜寐尚可，舌红，苔黄，脉滑数。\n患者主诉：\n肛旁肿痛3月。\n患者四诊信息：\n神志清晰，精神尚可，形体形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌红，苔黄，脉滑数。"
    result = medical_diagnosis_pipeline(test_input)
    print(f"\n🎯 最终诊断结果:\n{result}")
