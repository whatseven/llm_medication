import sys
import os
import json
from openai import OpenAI

# 添加src路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.search.milvus_search_copy import search_similar_diseases
from src.model.doctor import diagnose
from src.model.config import MODELS

def compress_search_results(user_input: str, search_results: list, model_name: str = None) -> list:
    """
    使用大模型压缩和过滤搜索结果，仅保留与查询相关的内容
    
    Args:
        user_input (str): 用户查询
        search_results (list): 搜索结果列表
        model_name (str): 模型名称，默认使用deepseek
    
    Returns:
        list: 压缩后的相关疾病列表
    """
    # 使用deepseek模型
    if not model_name:
        model_name = "deepseek"
    
    model_config = MODELS[model_name]
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"]
    )
    
    # 将搜索结果格式化为文本
    documents_text = ""
    for i, result in enumerate(search_results, 1):
        documents_text += f"文档{i}:\n"
        documents_text += f"OID: {result['oid']}\n"
        documents_text += f"疾病名称: {result['name']}\n"
        documents_text += f"描述: {result['desc']}\n"
        documents_text += f"症状: {result['symptom']}\n"
        documents_text += f"相似度: {result['similarity_score']}\n\n"
    
    system_prompt = """你是一个信息过滤专家。
你的任务是分析文档片段,提取与用户查询相关信息，返回所有有助于回答的信息，删除无关内容。

你的输出应该:
1. 返回所有有助于回答查询的信息，宁可返回错误，不可溜掉需要的信息
2. 保持相关句子的原始措辞(不要改写)
3. 保持文本的原始顺序
4. 包含所有相关内容,即使看起来有些冗余
5. 排除所有与查询无关的文本
6. 返回5~10个疾病信息，如果有更多，返回所有
输出格式要求：
将相关疾病的OID放在<relevant_oids>标签中，格式为JSON数组：
<relevant_oids>["5bb578c3831b973a137e43b9", "5bb578fd831b973a137e5f5a"]</relevant_oids>

注意：
- 如果多个疾病都很相关，可以都选择
- 避免返回空列表，除非真的没有任何相关疾病"""
    
    user_prompt = f"用户查询: {user_input}\n\n文档内容:\n{documents_text}"
    
    try:
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        compressed_content = response.choices[0].message.content.strip()
        
        # 解析<relevant_oids>标签中的内容
        import re
        import json
        
        pattern = r'<relevant_oids>(.*?)</relevant_oids>'
        match = re.search(pattern, compressed_content, re.DOTALL)
        
        if match:
            oids_str = match.group(1).strip()
            try:
                # 尝试解析JSON数组
                selected_oids = json.loads(oids_str)
                if not isinstance(selected_oids, list):
                    selected_oids = []
            except json.JSONDecodeError:
                print(f"无法解析OID列表: {oids_str}")
                selected_oids = []
        else:
            print("未找到<relevant_oids>标签")
            selected_oids = []
        
        # 如果没有相关疾病，返回所有原始搜索结果
        if not selected_oids:
            print("未找到相关疾病OID，返回所有搜索结果")
            print(f"模型响应内容: {compressed_content[:200]}...")
            return search_results
        
        # 根据OID匹配原始搜索结果
        filtered_results = []
        for result in search_results:
            if result['oid'] in selected_oids:
                filtered_results.append(result)
        
        return filtered_results
        
    except Exception as e:
        print(f"压缩过程出错: {str(e)}")
        # 如果压缩失败，返回原始搜索结果
        return search_results

def contextual_compression_rag_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None) -> str:
    """
    基于上下文压缩的医疗诊断流程（Contextual Compression RAG）
    搜索更多结果，使用大模型压缩过滤，然后进行诊断
    
    Args:
        user_input (str): 用户输入的症状描述
        model_name (str): 使用的模型名称，可选
        disease_list_file (str): 疾病列表文件路径，可选
    
    Returns:
        str: 最终诊断结果
    """
    try:
        print("开始Contextual Compression RAG医疗诊断流程...")
        print(f"用户输入: {user_input}")
        
        # 步骤1: 向量搜索（获取更多结果）
        print("\n步骤1: 向量搜索（top-30）...")
        milvus_results = search_similar_diseases(user_input, top_k=25)
        print(f"搜索到 {len(milvus_results)} 个疾病")
        
        if not milvus_results:
            return "未找到相关疾病信息，请咨询专业医生。"
        
        # 步骤2: 上下文压缩
        print("\n步骤2: 上下文压缩和过滤...")
        filtered_results = compress_search_results(user_input, milvus_results, model_name)
        print(f"压缩后保留 {len(filtered_results)} 个相关疾病")
        
        # 步骤3: 最终诊断（基于压缩后的内容）
        print("\n步骤3: 最终诊断（基于压缩内容）...")
        # 直接使用压缩后的疾病列表进行诊断
        diagnosis_result = diagnose(user_input, filtered_results, {}, model_name, disease_list_file)
        
        print("\nContextual Compression RAG诊断完成!")
        return diagnosis_result
        
    except Exception as e:
        error_msg = f"Contextual Compression RAG诊断流程出错: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    # 示例调用
    test_input = "我最近腹泻很严重，伴有腹痛，肛门也很疼"
    result = contextual_compression_rag_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
    
    print("\n" + "="*60)
    print("对比测试：复杂病例")
    
    # 复杂病例测试
    complex_test_input = "患者：从星期一喝完酒了，之后就一直拉肚子就是那种肚子疼，然后拉的时候，肛门痛，前两天特别痛，最近也一直在吃药吃了思密达和诺氟沙星胶囊但也老是不好，现在就是一直有时候拉的厉害点，有时候要好点"
    result2 = contextual_compression_rag_pipeline(complex_test_input)
    print(f"\n复杂病例诊断结果:\n{result2}")
