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
        print("警告: 无法导入OpenAI客户端，Self-RAG功能可能不可用")
        OpenAI = None


def judge_retrieval_need(user_input: str, model_name: str = None) -> bool:
    """
    判断是否需要进行信息检索
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
    
    Returns:
        bool: True表示需要检索，False表示不需要检索
    """
    try:
        # 检查OpenAI客户端是否可用
        if OpenAI is None:
            print("警告: OpenAI客户端不可用，默认需要检索")
            return True
            
        # 使用默认模型配置
        if model_name is None:
            model_name = "deepseek"
        
        model_config = MODELS[model_name]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        prompt = f"""你是一位专业的医疗诊断专家。请判断基于以下患者症状描述，是否需要从医疗知识库中检索相关疾病信息来辅助诊断。

患者症状描述：
{user_input}

请将你的判断结果用标签包围：
<decision>需要检索</decision> 或 <decision>不需要检索</decision>
"""

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位专业的医疗诊断专家，擅长判断诊断过程中是否需要额外的医疗信息支持。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"检索需求判断结果: {content}")
        
        # 提取判断结果
        pattern = r'<decision>(.*?)</decision>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            decision = match.group(1).strip()
            need_retrieval = "需要检索" in decision
            print(f"判断结果: {'需要检索' if need_retrieval else '不需要检索'}")
            return need_retrieval
        
        # 如果无法提取，默认需要检索
        print("警告: 无法提取判断结果，默认需要检索")
        return True
        
    except Exception as e:
        print(f"检索需求判断出错: {str(e)}，默认需要检索")
        return True


def self_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    基于自我反思的医疗诊断流程（Self-RAG）
    先判断是否需要检索，然后条件性地进行向量搜索和诊断
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("=== 开始Self-RAG医疗诊断流程 ===")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 判断是否需要检索
        print("\n步骤1: 判断是否需要信息检索...")
        need_retrieval = judge_retrieval_need(user_input, model_name)
        
        if need_retrieval:
            print("判断结果: 需要检索 - 进行向量库搜索")
            
            # 步骤2: 向量库搜索
            print("\n步骤2: 向量库搜索...")
            milvus_results = search_similar_diseases(user_input, top_k=5)
            print(f"搜索到 {len(milvus_results)} 个疾病")
            
            if not milvus_results:
                print("向量搜索无结果，使用空结果进行诊断")
                milvus_results = []
            
        else:
            print("判断结果: 不需要检索 - 直接进行诊断")
            milvus_results = []
        
        # 步骤3: 最终诊断
        print(f"\n步骤3: 最终诊断（{'基于搜索结果' if need_retrieval and milvus_results else '直接诊断'}）...")
        # 传入空的graph_data字典，表示不使用图数据库信息
        diagnosis_result = diagnose(user_input, milvus_results, {}, model_name, disease_list_file)
        
        print("\nSelf-RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"Self-RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    # 示例调用
    print("=== Self-RAG测试 ===\n")
    
    # 测试1: 复杂症状（应该需要检索）
    test_input1 = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    print(f"测试1: {test_input1}")
    result1 = self_rag_pipeline(test_input1)
    print(f"\n诊断结果1:\n{result1}")
    
    print("\n" + "="*80)
    
    # 测试2: 复杂病例（应该需要检索）
    test_input2 = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    print(f"测试2: {test_input2}")
    result2 = self_rag_pipeline(test_input2)
    print(f"\n诊断结果2:\n{result2}")
    
    print("\n" + "="*80)
    
    # 测试3: 简单症状（可能不需要检索）
    test_input3 = "我感觉有点累"
    print(f"测试3: {test_input3}")
    result3 = self_rag_pipeline(test_input3)
    print(f"\n诊断结果3:\n{result3}")
    
    print("\n" + "="*80)
    
    # 测试4: 明确症状（应该需要检索）
    test_input4 = "我最近总是感觉头晕目眩，特别是站起来的时候，有时候还会耳鸣"
    print(f"测试4: {test_input4}")
    result4 = self_rag_pipeline(test_input4)
    print(f"\n诊断结果4:\n{result4}")
