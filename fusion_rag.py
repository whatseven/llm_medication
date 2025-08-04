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
        print("警告: 无法导入OpenAI客户端，Fusion RAG功能可能不可用")
        OpenAI = None


def generate_medical_questions(user_input: str, model_name: str = None) -> list:
    """
    基于用户输入生成3个相关的医疗问题
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
    
    Returns:
        list: 生成的3个问题列表
    """
    try:
        # 检查OpenAI客户端是否可用
        if OpenAI is None:
            print("警告: OpenAI客户端不可用，无法生成问题")
            return [user_input, user_input, user_input]  # 回退到原始输入
            
        # 使用默认模型配置
        if model_name is None:
            model_name = "deepseek"
        
        model_config = MODELS[model_name]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        prompt = f"""你是一位专业的医疗诊断专家。基于患者的症状描述，请生成3个不同角度的医疗相关问题，这些问题将用于在医疗知识库中搜索相关疾病信息。

患者症状描述：
{user_input}

请从以下3个不同角度生成问题：
1. 主要症状相关的疾病
2. 伴随症状可能指向的疾病  
3. 症状组合可能的诊断方向

请将每个问题用标签包围，格式如下：
<question1>第一个问题</question1>
<question2>第二个问题</question2>
<question3>第三个问题</question3>

要求：
- 问题要具体明确，便于检索
- 覆盖不同的诊断角度
- 语言简洁专业"""

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位专业的医疗诊断专家，擅长从患者症状中提炼关键医疗问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"LLM生成的问题内容: {content}")
        
        # 提取3个问题
        questions = []
        for i in range(1, 4):
            pattern = f'<question{i}>(.*?)</question{i}>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                question = match.group(1).strip()
                questions.append(question)
            else:
                # 如果提取失败，使用原始输入作为备选
                questions.append(user_input)
        
        # 确保返回3个问题
        while len(questions) < 3:
            questions.append(user_input)
        
        print(f"提取到的问题: {questions}")
        return questions[:3]  # 只返回前3个
        
    except Exception as e:
        print(f"生成问题出错: {str(e)}，使用原始输入作为备选")
        return [user_input, user_input, user_input]


def merge_and_deduplicate_results(search_results_list: list) -> list:
    """
    合并多次搜索结果并通过oid去重
    
    Args:
        search_results_list (list): 多次搜索结果的列表
    
    Returns:
        list: 去重后的合并结果
    """
    seen_oids = set()
    merged_results = []
    
    for search_results in search_results_list:
        for result in search_results:
            oid = result.get('oid')
            if oid and oid not in seen_oids:
                seen_oids.add(oid)
                merged_results.append(result)
    
    print(f"合并前总数: {sum(len(results) for results in search_results_list)}")
    print(f"去重后总数: {len(merged_results)}")
    
    return merged_results


def fusion_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    基于问题生成和多次向量搜索的融合医疗诊断流程（Fusion RAG）
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("=== 开始Fusion RAG医疗诊断流程 ===")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 生成3个相关问题
        print("\n步骤1: 生成医疗相关问题...")
        questions = generate_medical_questions(user_input, model_name)
        print(f"生成的3个问题:")
        for i, question in enumerate(questions, 1):
            print(f"  问题{i}: {question}")
        
        # 步骤2: 对每个问题进行向量搜索
        print("\n步骤2: 分别进行向量搜索...")
        all_search_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n搜索问题{i}: {question}")
            search_results = search_similar_diseases(question, top_k=5)
            print(f"问题{i}搜索到 {len(search_results)} 个结果")
            all_search_results.append(search_results)
        
        # 步骤3: 合并和去重
        print("\n步骤3: 合并搜索结果并去重...")
        merged_results = merge_and_deduplicate_results(all_search_results)
        
        if not merged_results:
            return "未找到相关疾病信息，请咨询专业医生。"
        
        print(f"最终获得 {len(merged_results)} 个去重后的疾病信息")
        
        # 步骤4: 最终诊断
        print("\n步骤4: 基于融合结果进行诊断...")
        # 传入空的graph_data字典，表示不使用图数据库信息
        diagnosis_result = diagnose(user_input, merged_results, {}, model_name, disease_list_file)
        
        print("\nFusion RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"Fusion RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    # 示例调用
    print("=== Fusion RAG测试 ===\n")
    
    # 测试1: 基础症状
    test_input1 = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    print(f"测试1: {test_input1}")
    result1 = fusion_rag_pipeline(test_input1)
    print(f"\n诊断结果1:\n{result1}")
    
    print("\n" + "="*80)
    
    # 测试2: 复杂病例
    test_input2 = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    print(f"测试2: {test_input2}")
    result2 = fusion_rag_pipeline(test_input2)
    print(f"\n诊断结果2:\n{result2}")
    
    print("\n" + "="*80)
    
    # 测试3: 简单症状
    test_input3 = "我最近总是感觉头晕目眩，特别是站起来的时候"
    print(f"测试3: {test_input3}")
    result3 = fusion_rag_pipeline(test_input3)
    print(f"\n诊断结果3:\n{result3}")
