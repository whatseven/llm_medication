import json
import sys
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from openai import OpenAI

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import CRITICAL_DOCTOR_EVALUATION_PROMPT

def load_evaluation_results(file_path: str) -> List[Dict[str, Any]]:
    """加载评估结果数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip() and 
                json.loads(line.strip()).get('status') == 'success']

def extract_evaluation_result(response_text: str) -> int:
    """从大模型响应中提取评估结果"""
    try:
        patterns = [r'<r>\s*([01])\s*</r>', r'<r>([01])</r>', 
                   r'result[：:]\s*([01])', r'评估结果[：:]\s*([01])']
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match and match.group(1) in ['0', '1']:
                return int(match.group(1))
        return -1
    except Exception:
        return -1

def call_llm_evaluation(item: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> str:
    """调用大模型进行诊断质量评估"""
    try:
        model_config = MODELS[model_name]
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # 格式化真实疾病标签
        ground_truth_str = '、'.join(item['ground_truth_disease']) if isinstance(item['ground_truth_disease'], list) else str(item['ground_truth_disease'])
        
        prompt = CRITICAL_DOCTOR_EVALUATION_PROMPT.format(
            input_dialog=item['input_text'],
            ground_truth_disease=ground_truth_str,
            predicted_diseases=item['predicted_diseases'],
            raw_diagnosis=item['raw_diagnosis']
        )
        
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"大模型调用失败: {str(e)}")

def process_single_evaluation(item: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """处理单个评估项"""
    try:
        start_time = time.time()
        llm_response = call_llm_evaluation(item, model_name)
        evaluation_result = extract_evaluation_result(llm_response)
        
        result = {
            'id': item['id'],
            'original_ground_truth': item['ground_truth_disease'],
            'model_prediction': item['predicted_diseases'],
            'input_text': item['input_text'],
            'raw_diagnosis': item['raw_diagnosis'],
            'use_context': item.get('use_context', False),
            'llm_evaluation_result': evaluation_result,
            'llm_evaluation_reasoning': llm_response,
            'evaluation_time': round(time.time() - start_time, 2),
            'status': 'success' if evaluation_result != -1 else 'extract_failed'
        }
        
        status_symbol = "✓" if evaluation_result == 1 else ("✗" if evaluation_result == 0 else "⚠")
        print(f"{status_symbol} 完成评估ID {item['id']}: {evaluation_result}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} 评估失败: {str(e)}")
        return {
            'id': item['id'],
            'llm_evaluation_result': -1,
            'llm_evaluation_reasoning': f"处理错误: {str(e)}",
            'evaluation_time': 0,
            'status': 'error'
        }

def evaluate_diagnosis_quality(input_file: str, output_file: str, max_workers: int = 100, 
                             limit: int = None, model_name: str = DEFAULT_MODEL):
    """评估诊断质量"""
    print(f"开始RJUA诊断质量评估: {os.path.basename(input_file)}")
    print(f"使用模型: {model_name}")
    
    # 加载数据集
    dataset = load_evaluation_results(input_file)
    if limit:
        dataset = dataset[:limit]
    
    print(f"处理数据: {len(dataset)} 条")
    if len(dataset) == 0:
        print("没有可处理的数据")
        return []
    
    # 并发处理
    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_single_evaluation, item, model_name): item 
            for item in dataset
        }
        
        for future in as_completed(future_to_item):
            results.append(future.result())
    
    # 排序并统计
    results.sort(key=lambda x: int(x['id']))
    total_time = round(time.time() - start_time, 2)
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\n处理完成! 总耗时: {total_time}秒")
    print(f"成功评估: {success_count}, 处理错误: {error_count}")
    
    # 统计评估结果
    if success_count > 0:
        correct_count = sum(1 for r in results if r['llm_evaluation_result'] == 1)
        accuracy_rate = correct_count / success_count
        print(f"质量评估准确率: {accuracy_rate:.4f} ({correct_count}/{success_count})")
        
        # 按输入模式分析
        context_cases = [r for r in results if r.get('use_context', False) and r['status'] == 'success']
        question_only_cases = [r for r in results if not r.get('use_context', False) and r['status'] == 'success']
        
        if context_cases:
            context_correct = sum(1 for r in context_cases if r['llm_evaluation_result'] == 1)
            print(f"使用背景知识准确率: {context_correct/len(context_cases):.4f} ({context_correct}/{len(context_cases)})")
        
        if question_only_cases:
            question_correct = sum(1 for r in question_only_cases if r['llm_evaluation_result'] == 1)
            print(f"仅问题准确率: {question_correct/len(question_only_cases):.4f} ({question_correct}/{len(question_only_cases)})")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"结果保存到: {output_file}")
    return results

def analyze_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析评估结果"""
    total = len(results)
    success_results = [r for r in results if r['status'] == 'success']
    success_count = len(success_results)
    
    if success_count == 0:
        return {
            'total': total,
            'success_count': 0,
            'quality_accuracy': 0.0
        }
    
    # 统计质量评估结果
    correct_diagnoses = sum(1 for r in success_results if r['llm_evaluation_result'] == 1)
    incorrect_diagnoses = sum(1 for r in success_results if r['llm_evaluation_result'] == 0)
    
    # 按输入模式分析（如果有的话）
    context_cases = [r for r in success_results if r.get('use_context', False)]
    question_only_cases = [r for r in success_results if not r.get('use_context', False)]
    
    context_correct = sum(1 for r in context_cases if r['llm_evaluation_result'] == 1)
    question_only_correct = sum(1 for r in question_only_cases if r['llm_evaluation_result'] == 1)
    
    analysis = {
        'total': total,
        'success_count': success_count,
        'correct_diagnoses': correct_diagnoses,
        'incorrect_diagnoses': incorrect_diagnoses,
        'quality_accuracy': round(correct_diagnoses / success_count, 4) if success_count > 0 else 0.0,
        'context_cases_count': len(context_cases),
        'question_only_cases_count': len(question_only_cases),
        'context_accuracy': round(context_correct / len(context_cases), 4) if len(context_cases) > 0 else 0.0,
        'question_only_accuracy': round(question_only_correct / len(question_only_cases), 4) if len(question_only_cases) > 0 else 0.0
    }
    
    return analysis

if __name__ == "__main__":
    # ==================== 配置参数区域 ====================
    # 输入评估结果文件路径
    input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/final_result/rjua/contextual_compression/双向量字段/contextual_compression_rag_rjua_evaluation_results_top15.jsonl"
    
    # 输出目录和文件名
    output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/final_result/rjua/contextual_compression/双向量字段"
    output_file = os.path.join(output_dir, "rjua_quality.jsonl")
    
    # 评估模型配置
    model_name = DEFAULT_MODEL  # 使用默认模型
    # model_name = "qwen2.5:72b"  # 或指定其他模型
    # ====================================================
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行质量评估
    print("=== RJUA中文数据集诊断质量评估 ===")
    choice = input("选择评估模式:\n1. 测试模式(前5条)\n2. 小批量(前20条)\n3. 中批量(前50条)\n4. 全量评估\n请选择(1/2/3/4): ").strip()
    
    if choice == '1':
        limit = 5
        max_workers = 2
    elif choice == '2':
        limit = 20
        max_workers = 5
    elif choice == '3':
        limit = 50
        max_workers = 20
    elif choice == '4':
        limit = None
        max_workers = 50
    else:
        print("无效选择，使用测试模式")
        limit = 5
        max_workers = 2
    
    # 执行质量评估
    results = evaluate_diagnosis_quality(input_file, output_file, max_workers, limit, model_name)
    
    # 详细分析
    print("\n=== 详细质量分析 ===")
    analysis = analyze_evaluation_results(results)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print(f"\n结果已保存到: {output_file}") 