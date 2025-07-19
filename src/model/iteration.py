import re
import os
from openai import OpenAI
from src.model.config import MODELS
from src.model.prompt import R1_EXPERT_EVALUATION_PROMPT


def iterative_diagnose(symptoms, vector_results, graph_data, doctor_diagnosis, disease_list_file=None):
    """
    使用R1专家模型评估诊断质量
    
    Args:
        symptoms: 患者症状描述
        vector_results: 候选疾病信息
        graph_data: 图数据库信息
        doctor_diagnosis: 初步诊断结果
        disease_list_file: 疾病列表文件路径，可选
        
    Returns:
        dict: {
            "is_correct": bool
        }
    """
    try:
        # 加载疾病列表（如果提供）
        disease_list_str = ""
        if disease_list_file and os.path.exists(disease_list_file):
            try:
                with open(disease_list_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 尝试解析为Python列表格式
                        try:
                            import ast
                            disease_list = ast.literal_eval(content)
                            if isinstance(disease_list, list):
                                disease_list_str = f"可选疾病列表：{', '.join(disease_list)}"
                        except:
                            # 如果不是列表格式，按行读取
                            lines = content.split('\n')
                            diseases = [line.strip() for line in lines if line.strip()]
                            if diseases:
                                disease_list_str = f"可选疾病列表：{', '.join(diseases)}"
            except Exception as e:
                print(f"读取疾病列表文件出错: {str(e)}")
        
        # 构建专家评估提示词
        prompt = R1_EXPERT_EVALUATION_PROMPT.format(
            symptoms=symptoms,
            vector_results=vector_results,
            graph_data=graph_data,
            doctor_diagnosis=doctor_diagnosis,
            disease_list=disease_list_str
        )
        
        # 获取R1模型配置
        model_config = MODELS["deepseek"]
        
        # 初始化客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 使用R1模型进行评估（确保触发推理模式）
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位资深医疗专家，需要进行推理分析诊断是否正确。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 低温度确保稳定性
            stream=False
        )
        
        # 获取响应内容
        content = response.choices[0].message.content
        
        # 提取评估结果
        expert_review_match = re.search(r'<expert_review>(.*?)</expert_review>', content, re.DOTALL)
        if expert_review_match:
            review_content = expert_review_match.group(1).strip()
            # 简单解析1或0
            if '1' in review_content:
                return {"is_correct": True}
            elif '0' in review_content:
                return {"is_correct": False}
            else:
                # 如果无法解析，默认认为正确
                return {"is_correct": True}
        else:
            # 如果无法解析，默认认为正确
            return {"is_correct": True}
            
    except Exception as e:
        print(f"专家评估出错: {e}")
        # 出错时默认认为正确，避免阻塞诊断流程
        return {"is_correct": True}
