import sys
import os
import json
import re

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.search.milvus_search_copy import search_similar_diseases
from src.model.doctor import diagnose
from src.model.config import MODELS

# 导入LLM客户端
try:
    from openai import OpenAI
except ImportError:
    try:
        from lightrag.llm import OpenAIClient as OpenAI
    except ImportError:
        print("警告: 无法导入OpenAI客户端，HyDE RAG功能可能不可用")
        OpenAI = None


def generate_hypothetical_document(user_input: str, model_name: str = None) -> str:
    """
    基于用户输入生成假设性医疗文档
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
    
    Returns:
        str: 生成的假设性文档
    """
    try:
        # 检查OpenAI客户端是否可用
        if OpenAI is None:
            print("警告: OpenAI客户端不可用，使用原始输入作为假设性文档")
            return user_input
            
        # 使用默认模型配置
        if model_name is None:
            model_name = "deepseek"
        
        model_config = MODELS[model_name]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        prompt = f"""你是一位专业的医疗诊断专家。基于患者的症状描述，请生成一份假设性的医疗文档，这份文档应该详细描述可能的疾病、症状特征、诊断要点等信息，用于在医疗知识库中进行相似性搜索。

患者症状描述：
{user_input}

请生成一份假设性的医疗文档，包含：
1. 可能的疾病名称和描述
2. 相关的症状特征和表现
3. 诊断要点和鉴别诊断
4. 相关的医学术语和描述

请将生成的假设性文档用标签包围：
<document>
假设性医疗文档内容
</document>

要求：
- 文档内容要专业准确
- 包含丰富的医学描述
- 语言简洁，300字以内"""

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位专业的医疗诊断专家，擅长基于症状生成详细的医疗文档。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"LLM生成的假设性文档内容: {content}")
        
        # 提取假设性文档
        pattern = r'<document>(.*?)</document>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            hypothetical_doc = match.group(1).strip()
            print(f"提取到的假设性文档: {hypothetical_doc}")
            return hypothetical_doc
        else:
            # 如果提取失败，使用原始输入作为备选
            print("警告: 无法提取假设性文档，使用原始输入")
            return user_input
        
    except Exception as e:
        print(f"生成假设性文档出错: {str(e)}，使用原始输入作为备选")
        return user_input


def hyde_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    基于假设性文档生成的医疗诊断流程（HyDE RAG）
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("=== 开始HyDE RAG医疗诊断流程 ===")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 生成假设性文档
        print("\n步骤1: 生成假设性医疗文档...")
        hypothetical_doc = generate_hypothetical_document(user_input, model_name)
        print(f"假设性文档: {hypothetical_doc[:200]}...")  # 显示前200字符
        
        # 步骤2: 使用假设性文档进行向量搜索
        print("\n步骤2: 基于假设性文档进行向量搜索...")
        milvus_results = search_similar_diseases(hypothetical_doc, top_k=5)
        print(f"搜索到 {len(milvus_results)} 个疾病")
        
        if not milvus_results:
            return "未找到相关疾病信息，请咨询专业医生。"
        
        # 步骤3: 最终诊断（基于向量库结果）
        print("\n步骤3: 基于搜索结果进行最终诊断...")
        # 传入空的graph_data字典，表示不使用图数据库信息
        # 注意：这里传入的是原始用户输入，而不是假设性文档，因为诊断时需要针对用户的实际症状
        diagnosis_result = diagnose(user_input, milvus_results, {}, model_name, disease_list_file)
        
        print("\nHyDE RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"HyDE RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    # 示例调用
    print("=== HyDE RAG测试 ===\n")
    
    # 测试1: 基础症状
    test_input1 = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    print(f"测试1: {test_input1}")
    result1 = hyde_rag_pipeline(test_input1)
    print(f"\n诊断结果1:\n{result1}")
    
    print("\n" + "="*80)
    
    # 测试2: 复杂病例
    test_input2 = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    print(f"测试2: {test_input2}")
    result2 = hyde_rag_pipeline(test_input2)
    print(f"\n诊断结果2:\n{result2}")
    
    print("\n" + "="*80)
    
    # 测试3: 简单症状
    test_input3 = "我最近总是感觉头晕目眩，特别是站起来的时候"
    print(f"测试3: {test_input3}")
    result3 = hyde_rag_pipeline(test_input3)
    print(f"\n诊断结果3:\n{result3}")
