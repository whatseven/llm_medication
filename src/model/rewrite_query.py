from ..utils.rewrite import call_symptom_api, extract_symptoms_from_response

def process_dialog_symptoms(dialog_text, model_name=None):
    """
    完整的症状处理流程：提取症状并改写为专业术语
    
    Args:
        dialog_text (str): 对话文本
        model_name (str): 模型名称
    
    Returns:
        list: 症状列表
    """
    try:
        response = call_symptom_api(dialog_text, model_name)
        symptoms = extract_symptoms_from_response(response)
        return symptoms
    except Exception as e:
        raise Exception(f"症状提取失败: {str(e)}")
