import sys
import os
import json

# 设置博茶API密钥环境变量
os.environ['BOCHAAI_API_KEY'] = 'sk-458cada7190d42e0aff052c288ab05c5'

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.search.milvus_search_copy import search_similar_diseases
from src.search.web_search import web_search
from src.model.doctor import diagnose
from src.model.config import MODELS

# 使用lightrag环境的OpenAI客户端
try:
    from openai import OpenAI
except ImportError:
    # 如果openai库不可用，尝试使用lightrag的客户端
    try:
        from lightrag.llm import OpenAIClient as OpenAI
    except ImportError:
        print("警告: 无法导入OpenAI客户端，相关性评估功能可能不可用")
        OpenAI = None


def evaluate_relevance(user_input: str, vector_results: list, model_name: str = None) -> int:
    """
    评估向量搜索结果与用户输入的相关性
    
    Args:
        user_input (str): 用户输入的症状描述
        vector_results (list): 向量搜索返回的疾病列表
        model_name (str): 使用的模型名称，可选
    
    Returns:
        int: 相关性评分 (2=高度相关, 1=中度相关, 0=低度相关)
    """
    try:
        # 检查OpenAI客户端是否可用
        if OpenAI is None:
            print("警告: OpenAI客户端不可用，默认返回中度相关(1)")
            return 1
            
        # 使用默认模型配置
        if model_name is None:
            model_name = "deepseek"
        
        model_config = MODELS[model_name]
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 构建向量搜索结果的文本描述
        vector_text = ""
        for i, disease in enumerate(vector_results, 1):
            vector_text += f"{i}. {disease.get('name', 'Unknown')}\n"
            vector_text += f"   描述：{disease.get('desc', 'No description')}\n"
            vector_text += f"   症状：{disease.get('symptom', 'No symptoms')}\n"
            vector_text += f"   相似度：{disease.get('similarity_score', 0):.3f}\n\n"
        
        prompt = f"""你是一位医疗专家，需要评估向量搜索结果与患者症状的相关性。

患者症状描述：
{user_input}

向量搜索返回的疾病信息：
{vector_text}

请评估这些搜索结果与患者症状的相关性：
- 高度相关(2)：搜索到的疾病与患者症状高度匹配，症状描述详细准确
- 中度相关(1)：搜索到的疾病部分匹配患者症状，但可能需要更多信息补充
- 低度相关(0)：搜索到的疾病与患者症状匹配度较低，不太相关

请将评估结果放在<relevance>标签中：
<relevance>评分数字</relevance>"""

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位专业的医疗诊断专家，擅长评估医疗信息的相关性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            stream=False
        )
        
        content = response.choices[0].message.content
        print(f"相关性评估结果: {content}")
        
        # 提取相关性评分
        import re
        relevance_match = re.search(r'<relevance>(\d+)</relevance>', content)
        if relevance_match:
            relevance_score = int(relevance_match.group(1))
            if relevance_score in [0, 1, 2]:
                return relevance_score
        
        # 如果无法提取，默认返回中度相关
        print("警告: 无法提取相关性评分，默认返回中度相关(1)")
        return 1
        
    except Exception as e:
        print(f"相关性评估出错: {str(e)}，默认返回中度相关(1)")
        return 1


def format_web_search_results(web_results: dict) -> list:
    """
    格式化web搜索结果为统一格式
    
    Args:
        web_results (dict): web搜索API返回的结果
    
    Returns:
        list: 格式化后的搜索结果列表
    """
    formatted_results = []
    
    try:
        if web_results.get("code") == 200 and "data" in web_results:
            web_pages = web_results["data"].get("webPages", {}).get("value", [])
            
            for page in web_pages:
                formatted_result = {
                    "name": page.get("name", ""),
                    "desc": page.get("snippet", ""),
                    "symptom": "",  # web搜索结果没有专门的症状字段
                    "similarity_score": 0.9,  # 给web搜索结果一个默认高分
                    "source": "web_search",  # 标记信息来源
                    "url": page.get("url", "")
                }
                formatted_results.append(formatted_result)
        
    except Exception as e:
        print(f"格式化web搜索结果出错: {str(e)}")
    
    return formatted_results


def corrective_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    Corrective RAG医疗诊断流程
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("=== 开始Corrective RAG医疗诊断流程 ===")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 向量库搜索
        print("\n步骤1: 向量库搜索(top5)...")
        vector_results = search_similar_diseases(user_input, top_k=5)
        print(f"向量搜索到 {len(vector_results)} 个疾病")
        
        if not vector_results:
            print("向量搜索无结果，尝试web搜索...")
            try:
                web_results = web_search(user_input, count=5)
                formatted_web_results = format_web_search_results(web_results)
                if formatted_web_results:
                    return diagnose(user_input, formatted_web_results, {}, model_name, disease_list_file)
                else:
                    return "未找到相关疾病信息，请咨询专业医生。"
            except Exception as e:
                return f"搜索失败: {str(e)}"
        
        # 标记向量搜索结果的来源
        for result in vector_results:
            result["source"] = "vector_search"
        
        # 步骤2: 相关性评估
        print("\n步骤2: 相关性评估...")
        relevance_score = evaluate_relevance(user_input, vector_results, model_name)
        
        if relevance_score == 2:
            print("评估结果: 高度相关(2) - 直接使用向量库信息诊断")
            # 高度相关：直接使用向量库结果
            return diagnose(user_input, vector_results, {}, model_name, disease_list_file)
            
        elif relevance_score == 1:
            print("评估结果: 中度相关(1) - 结合向量库和web搜索信息")
            # 中度相关：结合向量库和web搜索
            try:
                print("\n步骤3: 执行web搜索补充信息...")
                web_results = web_search(user_input, count=5)
                formatted_web_results = format_web_search_results(web_results)
                print(f"Web搜索到 {len(formatted_web_results)} 个结果")
                
                # 合并向量库和web搜索结果
                combined_results = vector_results + formatted_web_results
                print(f"合并后共 {len(combined_results)} 个结果")
                
                return diagnose(user_input, combined_results, {}, model_name, disease_list_file)
                
            except Exception as e:
                print(f"Web搜索失败: {str(e)}，仅使用向量库信息")
                return diagnose(user_input, vector_results, {}, model_name, disease_list_file)
        
        else:  # relevance_score == 0
            print("评估结果: 低度相关(0) - 主要使用web搜索信息")
            # 低度相关：主要使用web搜索
            try:
                print("\n步骤3: 执行web搜索...")
                web_results = web_search(user_input, count=5)
                formatted_web_results = format_web_search_results(web_results)
                print(f"Web搜索到 {len(formatted_web_results)} 个结果")
                
                if formatted_web_results:
                    return diagnose(user_input, formatted_web_results, {}, model_name, disease_list_file)
                else:
                    print("Web搜索无有效结果，回退到向量库信息")
                    return diagnose(user_input, vector_results, {}, model_name, disease_list_file)
                    
            except Exception as e:
                print(f"Web搜索失败: {str(e)}，回退到向量库信息")
                return diagnose(user_input, vector_results, {}, model_name, disease_list_file)
        
    except Exception as e:
        error_msg = f"Corrective RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    # 示例调用
    print("=== Corrective RAG测试 ===\n")
    
    # 测试1: 基础症状
    test_input1 = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    print(f"测试1: {test_input1}")
    result1 = corrective_rag_pipeline(test_input1)
    print(f"\n诊断结果1:\n{result1}")
    
    print("\n" + "="*80)
    
    # 测试2: 复杂病例
    test_input2 = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    print(f"测试2: {test_input2}")
    result2 = corrective_rag_pipeline(test_input2)
    print(f"\n诊断结果2:\n{result2}")
    
    print("\n" + "="*80)
    
    # 测试3: 不常见症状
    test_input3 = "我最近总是感觉头晕目眩，特别是站起来的时候，有时候还会耳鸣"
    print(f"测试3: {test_input3}")
    result3 = corrective_rag_pipeline(test_input3)
    print(f"\n诊断结果3:\n{result3}")
