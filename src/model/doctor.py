import requests
import json
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import DOCTOR_SYSTEM_PROMPT

def diagnose(user_input: str, vector_results: list, graph_data: dict, model_name: str = DEFAULT_MODEL) -> str:
    """
    进行最终医疗诊断
    
    Args:
        user_input: 用户症状描述
        vector_results: 过滤后的向量库搜索结果
        graph_data: 图数据库查询的详细医学资料
        model_name: 使用的模型名称
    
    Returns:
        诊断结果文本
    """
    # 格式化向量库结果
    vector_info = ""
    for i, result in enumerate(vector_results, 1):
        vector_info += f"{i}. 疾病：{result.get('name', '')}\n"
        vector_info += f"   描述：{result.get('desc', '')}\n"
        vector_info += f"   症状：{result.get('symptom', '')}\n"
        vector_info += f"   相似度：{result.get('similarity_score', 0):.4f}\n\n"
    
    # 格式化图数据库结果
    graph_info = ""
    if graph_data:
        for key, value in graph_data.items():
            graph_info += f"{key}：{value}\n\n"
    
    # 手动替换占位符来避免与JSON格式冲突
    system_prompt = DOCTOR_SYSTEM_PROMPT.replace("{vector_results}", vector_info)
    system_prompt = system_prompt.replace("{graph_data}", graph_info)
    
    # 获取模型配置
    model_config = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_config["model_name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            f"{model_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        diagnosis_text = result["choices"][0]["message"]["content"]
        
        return diagnosis_text
        
    except Exception as e:
        return f"诊断过程中发生错误: {str(e)}"
