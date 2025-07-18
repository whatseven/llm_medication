import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import re

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from main import medical_diagnosis_pipeline

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集并添加id
    
    Args:
        file_path: 数据集文件路径
    
    Returns:
        添加了id的数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                item['id'] = idx  # 添加行号作为id
                data.append(item)
    return data

def preprocess_dialog(dialog: List[str]) -> str:
    """
    将对话列表转换为字符串
    
    Args:
        dialog: 对话列表
    
    Returns:
        拼接后的对话字符串
    """
    return '\n'.join(dialog)

def extract_diseases_from_diagnosis(diagnosis_text: str) -> List[str]:
    """
    从诊断结果中提取疾病名称
    根据1.md中的格式：<final_diagnosis>{"diseases": [产气杆菌肠炎]}</final_diagnosis>
    
    Args:
        diagnosis_text: 诊断结果文本
    
    Returns:
        提取到的疾病列表
    """
    try:
        # 查找<final_diagnosis>标签
        pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
        match = re.search(pattern, diagnosis_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            diagnosis_data = json.loads(json_str)
            diseases = diagnosis_data.get('diseases', [])
            return diseases if isinstance(diseases, list) else [diseases]
        
        # 如果没有找到标准格式，尝试其他方式提取
        # 查找常见的疾病诊断模式
        disease_patterns = [
            r'诊断[：:]\s*([^。\n]+)',
            r'可能的疾病[：:]\s*([^。\n]+)',
            r'初步诊断[：:]\s*([^。\n]+)',
            r'考虑[：:]?\s*([^。\n，,]+)',
        ]
        
        for pattern in disease_patterns:
            matches = re.findall(pattern, diagnosis_text)
            if matches:
                return [match.strip() for match in matches]
        
        return ["未能提取疾病信息"]
        
    except Exception as e:
        return [f"提取错误: {str(e)}"]

def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个数据项
    
    Args:
        item: 包含id和对话数据的字典
    
    Returns:
        处理结果字典
    """
    try:
        # 使用original_dialog
        dialog_text = preprocess_dialog(item['original_dialog'])
        
        # 调用诊断流程
        start_time = time.time()
        diagnosis_result = medical_diagnosis_pipeline(dialog_text)
        end_time = time.time()
        
        # 提取疾病信息
        predicted_diseases = extract_diseases_from_diagnosis(diagnosis_result)
        
        result = {
            'id': item['id'],
            'ground_truth_disease': item['disease'],
            'ground_truth_label': item['label'],
            'input_dialog': dialog_text,
            'raw_diagnosis': diagnosis_result,
            'predicted_diseases': predicted_diseases,
            'processing_time': round(end_time - start_time, 2),
            'status': 'success'
        }
        
        print(f"✓ 完成ID {item['id']}: {len(dialog_text[:50])}... -> {predicted_diseases}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} 处理失败: {str(e)}")
        return {
            'id': item['id'],
            'ground_truth_disease': item['disease'],
            'ground_truth_label': item['label'],
            'input_dialog': preprocess_dialog(item['original_dialog']),
            'raw_diagnosis': f"处理错误: {str(e)}",
            'predicted_diseases': ["处理失败"],
            'processing_time': 0,
            'status': 'error'
        }

def evaluate_dataset(input_file: str, output_file: str, max_workers: int = 5, limit: int = None):
    """
    评估整个数据集
    
    Args:
        input_file: 输入数据集文件路径
        output_file: 输出结果文件路径
        max_workers: 并发线程数
        limit: 限制处理的数据条数，None表示处理全部
    """
    print(f"开始评估数据集: {input_file}")
    print(f"并发线程数: {max_workers}")
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset(input_file)
    
    if limit:
        dataset = dataset[:limit]
        print(f"限制处理前 {limit} 条数据")
    
    print(f"总共 {len(dataset)} 条数据")
    
    # 并发处理
    print("\n开始并发处理...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_single_item, item): item for item in dataset}
        
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
    
    # 统计成功失败数
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("评估完成!")
    return results

def simple_accuracy_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    简单的准确率分析
    
    Args:
        results: 评估结果列表
    
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
            'error_count': total,
            'accuracy': 0.0
        }
    
    # 简单匹配：预测疾病与真实疾病有交集即认为正确
    correct_predictions = 0
    for result in success_results:
        predicted = set(result['predicted_diseases'])
        ground_truth = set(result['ground_truth_disease'])
        
        # 如果有交集，认为预测正确
        if predicted & ground_truth:
            correct_predictions += 1
    
    accuracy = correct_predictions / success_count if success_count > 0 else 0.0
    
    return {
        'total': total,
        'success_count': success_count,
        'error_count': total - success_count,
        'correct_predictions': correct_predictions,
        'accuracy': round(accuracy, 4)
    }

if __name__ == "__main__":
    # 配置文件路径
    input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/DiaMed/test.txt"
    output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/evaluation/llm_medication_rag/DD"
    output_file = os.path.join(output_dir, "evaluation_results.jsonl")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估（测试模式：只处理前10条）
    print("=== DiaMed数据集评估 ===")
    choice = input("选择模式:\n1. 测试模式(前10条)\n2. 小批量(前50条)\n3. 全量评估\n请选择(1/2/3): ").strip()
    
    if choice == '1':
        limit = 10
        max_workers = 3
    elif choice == '2':
        limit = 50
        max_workers = 5
    elif choice == '3':
        limit = None
        max_workers = 5
    else:
        print("无效选择，使用测试模式")
        limit = 10
        max_workers = 3
    
    # 执行评估
    results = evaluate_dataset(input_file, output_file, max_workers, limit)
    
    # 简单分析
    print("\n=== 简单准确率分析 ===")
    analysis = simple_accuracy_analysis(results)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print(f"\n结果已保存到: {output_file}")
