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
    """
    加载TCM评估结果数据集
    
    Args:
        file_path: 评估结果文件路径
    
    Returns:
        评估结果数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # 只处理成功的案例
                if item.get('status') == 'success':
                    data.append(item)
    return data

def extract_evaluation_result(response_text: str) -> int:
    """
    从大模型响应中提取评估结果
    
    Args:
        response_text: 大模型的完整响应文本
    
    Returns:
        评估结果：1表示正确，0表示错误，-1表示提取失败
    """
    try:
        # 查找<r>标签
        pattern = r'<r>\s*([01])\s*</r>'
        match = re.search(pattern, response_text, re.IGNORECASE)
        
        if match:
            result = int(match.group(1))
            return result
        
        # 备选提取模式
        backup_patterns = [
            r'<r>\s*(\d+)\s*</r>',
            r'<r>([01])</r>',            # 无空格版本
            r'result[：:]\s*([01])',
            r'评估结果[：:]\s*([01])',
            r'最终结果[：:]\s*([01])'
        ]
        
        for pattern in backup_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result = int(match.group(1))
                if result in [0, 1]:
                    return result
        
        return -1  # 提取失败
        
    except Exception as e:
        print(f"提取评估结果时出错: {str(e)}")
        return -1

def call_llm_evaluation(item: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> str:
    """
    调用大模型进行TCM诊断质量评估
    
    Args:
        item: 包含TCM诊断信息的数据项
        model_name: 使用的模型名称
    
    Returns:
        大模型的评估响应
    """
    try:
        # 获取模型配置
        model_config = MODELS[model_name]
        
        # 初始化客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 格式化提示词 - 适配TCM数据格式
        prompt = CRITICAL_DOCTOR_EVALUATION_PROMPT.format(
            input_dialog=item['input_text'],  # TCM使用拼接后的input_text
            ground_truth_disease=item['ground_truth_disease'],
            predicted_diseases=item['predicted_diseases'],
            raw_diagnosis=item['raw_diagnosis']
        )
        
        # 调用大模型
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"大模型调用失败: {str(e)}")

def process_single_evaluation(item: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    处理单个TCM评估项
    
    Args:
        item: 包含TCM诊断信息的数据项
        model_name: 使用的模型名称
    
    Returns:
        处理结果字典
    """
    try:
        # 调用大模型评估
        start_time = time.time()
        llm_response = call_llm_evaluation(item, model_name)
        end_time = time.time()
        
        # 提取评估结果
        evaluation_result = extract_evaluation_result(llm_response)
        
        result = {
            'id': item['id'],
            'original_ground_truth': item['ground_truth_disease'],
            'patient_info': item.get('patient_info', ''),
            'first_question': item.get('first_question', ''),
            'model_prediction': item['predicted_diseases'],
            'input_text': item['input_text'],
            'raw_diagnosis': item['raw_diagnosis'],
            'llm_evaluation_result': evaluation_result,
            'llm_evaluation_reasoning': llm_response,
            'evaluation_time': round(end_time - start_time, 2),
            'status': 'success' if evaluation_result != -1 else 'extract_failed'
        }
        
        status_symbol = "✓" if evaluation_result == 1 else ("✗" if evaluation_result == 0 else "⚠")
        print(f"{status_symbol} 完成TCM评估ID {item['id']}: {evaluation_result}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} TCM评估失败: {str(e)}")
        return {
            'id': item['id'],
            'original_ground_truth': item['ground_truth_disease'],
            'patient_info': item.get('patient_info', ''),
            'first_question': item.get('first_question', ''),
            'model_prediction': item['predicted_diseases'],
            'input_text': item['input_text'],
            'raw_diagnosis': item['raw_diagnosis'],
            'llm_evaluation_result': -1,
            'llm_evaluation_reasoning': f"处理错误: {str(e)}",
            'evaluation_time': 0,
            'status': 'error'
        }

def evaluate_tcm_diagnosis_quality(input_file: str, output_file: str, max_workers: int = 100, 
                                 limit: int = None, model_name: str = DEFAULT_MODEL):
    """
    评估TCM诊断质量
    
    Args:
        input_file: 输入的TCM评估结果文件路径
        output_file: 输出的质量评估结果文件路径
        max_workers: 并发线程数
        limit: 限制处理的数据条数，None表示处理全部
        model_name: 使用的模型名称
    """
    print(f"开始TCM诊断质量评估: {input_file}")
    print(f"使用模型: {model_name}")
    print(f"并发线程数: {max_workers}")
    
    # 加载数据集
    print("加载TCM评估结果数据...")
    dataset = load_evaluation_results(input_file)
    
    if limit:
        dataset = dataset[:limit]
        print(f"限制处理前 {limit} 条数据")
    
    print(f"总共 {len(dataset)} 条成功的TCM评估数据")
    
    if len(dataset) == 0:
        print("没有可处理的TCM数据")
        return []
    
    # 并发处理
    print("\n开始并发TCM质量评估...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_evaluation, item, model_name): item 
            for item in dataset
        }
        
        # 收集结果
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
    
    # 按id排序确保顺序正确
    results.sort(key=lambda x: x['id'])
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    print(f"\n处理完成! 总耗时: {total_time}秒")
    print(f"平均每条耗时: {round(total_time/len(dataset), 2)}秒")
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    extract_failed_count = sum(1 for r in results if r['status'] == 'extract_failed')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"成功评估: {success_count}")
    print(f"提取失败: {extract_failed_count}")
    print(f"处理错误: {error_count}")
    
    # 统计评估结果
    if success_count > 0:
        correct_count = sum(1 for r in results if r['llm_evaluation_result'] == 1)
        incorrect_count = sum(1 for r in results if r['llm_evaluation_result'] == 0)
        accuracy_rate = correct_count / success_count
        
        print(f"\n=== TCM质量评估统计 ===")
        print(f"诊断正确: {correct_count}")
        print(f"诊断错误: {incorrect_count}")
        print(f"准确率: {accuracy_rate:.4f} ({accuracy_rate*100:.2f}%)")
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("TCM质量评估完成!")
    return results

def analyze_tcm_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析TCM评估结果
    
    Args:
        results: TCM评估结果列表
    
    Returns:
        分析统计信息
    """
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
    
    # 统计不同证型的表现
    disease_types = {}
    for result in success_results:
        ground_truth = result['original_ground_truth']
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            disease_name = ground_truth[0]
        else:
            disease_name = str(ground_truth)
        
        if disease_name not in disease_types:
            disease_types[disease_name] = {'total': 0, 'correct': 0}
        
        disease_types[disease_name]['total'] += 1
        if result['llm_evaluation_result'] == 1:
            disease_types[disease_name]['correct'] += 1
    
    return {
        'total': total,
        'success_count': success_count,
        'correct_diagnoses': correct_diagnoses,
        'incorrect_diagnoses': incorrect_diagnoses,
        'quality_accuracy': round(correct_diagnoses / success_count, 4) if success_count > 0 else 0.0,
        'disease_types_count': len(disease_types),
        'disease_types_detail': disease_types
    }


if __name__ == "__main__":
    # 配置文件路径
    input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/final_result/MM/graph_rag/graph_rag_tcm_evaluation_results_top5.jsonl"
    output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/final_result/MM/graph_rag"
    output_file = os.path.join(output_dir, "mm_quality_evaluation_results.jsonl")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行质量评估
    print("=== TCM中医诊断质量评估 ===")
    print("选择评估模式:")
    print("1. 测试模式(前5条)")
    print("2. 小批量(前20条)")  
    print("3. 中批量(前50条)")
    print("4. 全量评估")
    
    choice = input("请选择(1/2/3/4): ").strip()
    
    if choice == '1':
        limit = 5
        max_workers = 2
    elif choice == '2':
        limit = 20
        max_workers = 5
    elif choice == '3':
        limit = 50
        max_workers = 100
    elif choice == '4':
        limit = None
        max_workers = 100
    else:
        print("无效选择，使用测试模式")
        limit = 5
        max_workers = 2
    
    # 选择模型
    print(f"\n可用模型: {list(MODELS.keys())}")
    model_choice = input(f"请选择模型 (默认: {DEFAULT_MODEL}): ").strip()
    model_name = model_choice if model_choice in MODELS else DEFAULT_MODEL
    
    # 执行质量评估
    results = evaluate_tcm_diagnosis_quality(input_file, output_file, max_workers, limit, model_name)
    
    # 检查是否有结果可供分析
    if not results or len(results) == 0:
        print("\n❌ 没有评估结果可供分析")
        exit(1)
    
    print(f"\n📊 获得 {len(results)} 条评估结果，开始分析...")
    
    # 详细分析
    print("\n=== 详细TCM质量分析 ===")
    try:
        analysis = analyze_tcm_evaluation_results(results)
        print(f"✅ 分析完成，获得 {len(analysis)} 项统计指标")
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        print(f"结果示例: {results[0] if results else 'No results'}")
        exit(1)
    
    # 首先显示总体准确率
    total_accuracy = analysis.get('quality_accuracy', 0.0)
    correct_count = analysis.get('correct_diagnoses', 0)
    total_count = analysis.get('success_count', 0)
    
    print(f"\n🎯 【TCM诊断总体准确率】")
    print(f"=" * 40)
    print(f"评估为正确的诊断: {correct_count}")
    print(f"成功评估的总数据: {total_count}")
    print(f"📊 总体准确率: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
    print(f"=" * 40)
    
    # 显示详细统计信息
    for key, value in analysis.items():
        if key == 'disease_types_detail':
            print(f"\n=== 各证型诊断表现 ===")
            for disease, stats in value.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{disease}: {stats['correct']}/{stats['total']} ({accuracy:.4f})")
        elif key not in ['quality_accuracy', 'correct_diagnoses', 'success_count']:  # 避免重复显示
            print(f"{key}: {value}")
    
    print(f"\n结果已保存到: {output_file}")