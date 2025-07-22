import json
import sys
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from main_textgrad import medical_diagnosis_pipeline

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载RJUA数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def parse_disease_labels(disease_str: str) -> List[str]:
    """解析疾病标签字符串，按中文标点符号分割"""
    if not disease_str:
        return []
    diseases = re.split(r'[、，,;；/\s]+', disease_str.strip())
    return [d.strip() for d in diseases if d.strip()]

def extract_diseases_from_diagnosis(diagnosis_text: str) -> List[str]:
    """从诊断结果中提取疾病名称"""
    try:
        # 查找<final_diagnosis>标签
        pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
        match = re.search(pattern, diagnosis_text, re.DOTALL)
        
        if match:
            diagnosis_data = json.loads(match.group(1))
            diseases = diagnosis_data.get('diseases', [])
            return diseases if isinstance(diseases, list) else [diseases]
        
        # 备选提取模式
        for pattern in [r'诊断[：:]\s*([^。\n]+)', r'可能的疾病[：:]\s*([^。\n]+)', 
                       r'初步诊断[：:]\s*([^。\n]+)', r'考虑[：:]?\s*([^。\n，,]+)']:
            matches = re.findall(pattern, diagnosis_text)
            if matches:
                return [match.strip() for match in matches]
        
        return ["未能提取疾病信息"]
    except Exception as e:
        return [f"提取错误: {str(e)}"]

def process_single_item(item: Dict[str, Any], disease_list_file: str = None, use_context: bool = False) -> Dict[str, Any]:
    """处理单个数据项"""
    try:
        # 预处理输入
        if use_context:
            input_text = f"患者问题：{item['question']}\n\n相关医学知识：\n{item['context']}"
        else:
            input_text = item['question']
        
        # 解析真实疾病标签
        ground_truth_diseases = parse_disease_labels(item['disease'])
        
        # 调用诊断流程
        start_time = time.time()
        diagnosis_result = medical_diagnosis_pipeline(input_text, disease_list_file=disease_list_file, silent_mode=True)
        end_time = time.time()
        
        # 提取预测疾病信息
        predicted_diseases = extract_diseases_from_diagnosis(diagnosis_result)
        
        result = {
            'id': item['id'],
            'ground_truth_disease': ground_truth_diseases,
            'ground_truth_answer': item['answer'],
            'ground_truth_advice': item['advice'],
            'input_text': input_text,
            'raw_diagnosis': diagnosis_result,
            'predicted_diseases': predicted_diseases,
            'processing_time': round(end_time - start_time, 2),
            'status': 'success',
            'use_context': use_context
        }
        
        print(f"✓ 完成ID {item['id']}: {predicted_diseases} vs {ground_truth_diseases}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} 处理失败: {str(e)}")
        return {
            'id': item['id'],
            'ground_truth_disease': parse_disease_labels(item['disease']),
            'input_text': item['question'],
            'raw_diagnosis': f"处理错误: {str(e)}",
            'predicted_diseases': ["处理失败"],
            'processing_time': 0,
            'status': 'error',
            'use_context': use_context
        }

def evaluate_dataset(input_file: str, output_file: str, max_workers: int = 100, 
                    limit: int = None, disease_list_file: str = None, use_context: bool = False):
    """评估整个RJUA数据集"""
    print(f"开始评估RJUA数据集: {os.path.basename(input_file)}")
    print(f"输入模式: {'问题+知识背景' if use_context else '仅问题'}")
    print(f"疾病列表约束: {'是' if disease_list_file else '否'}")
    
    # 加载数据集
    dataset = load_dataset(input_file)
    if limit:
        dataset = dataset[:limit]
    print(f"处理数据: {len(dataset)} 条")
    
    # 并发处理
    start_time = time.time()
    results = []
    completed_count = 0
    total_count = len(dataset)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, disease_list_file, use_context): item 
            for item in dataset
        }
        
        for future in as_completed(future_to_item):
            results.append(future.result())
            completed_count += 1
            
            # 每完成5个样本显示一次进度
            if completed_count % 5 == 0 or completed_count == total_count:
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / completed_count
                remaining_items = total_count - completed_count
                estimated_remaining_time = avg_time_per_item * remaining_items
                
                print(f"📊 进度: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%)")
                print(f"⏱️  已用时: {elapsed_time/60:.1f}分钟, 预计剩余: {estimated_remaining_time/60:.1f}分钟")
                print("=" * 50)
    
    # 排序并统计
    results.sort(key=lambda x: int(x['id']))
    total_time = round(time.time() - start_time, 2)
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"\n处理完成! 总耗时: {total_time}秒, 成功: {success_count}/{len(results)}")
    
    # 简单准确率分析
    if success_count > 0:
        correct = sum(1 for r in results if r['status'] == 'success' and 
                     set(r['predicted_diseases']) & set(r['ground_truth_disease']))
        accuracy = correct / success_count
        print(f"简单准确率: {accuracy:.4f} ({correct}/{success_count})")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"结果保存到: {output_file}")
    return results

def simple_accuracy_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """简单的准确率分析"""
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
    
    # 集合匹配：预测疾病与真实疾病有交集即认为正确
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
    # ==================== 配置参数区域 ====================
    # 输入数据集文件路径
    input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/RJUA_CN/RJUA_test.json"
    
    # 输出目录和文件名
    output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/RJUACN"
    output_file = os.path.join(output_dir, "evaluation_results4.jsonl")
    
    # 疾病列表文件路径配置（可选）
    # 设置为 None 表示不使用疾病列表约束
    # 设置为文件路径表示使用疾病列表约束
    disease_list_file ="/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/RJUA_CN/disease.txt"  # 默认不使用疾病列表约束
    # disease_list_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/RJUA_CN/disease.txt"  # 使用疾病列表约束
    
    # 输入模式配置
    use_context = False  # False: 仅使用问题, True: 使用问题+知识背景
    # ====================================================
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    print("=== RJUA中文数据集评估 ===")
    choice = input("选择模式:\n1. 测试模式(前10条)\n2. 小批量(前50条)\n3. 全量评估\n请选择(1/2/3): ").strip()
    
    if choice == '1':
        limit = 10
        max_workers = 2  # 减少并发数
        print("⚠️  测试模式：每个样本需约11次LLM调用，预计需要2-5分钟")
    elif choice == '2':
        limit = 50
        max_workers = 2  # 减少并发数
        print("⚠️  小批量模式：预计需要15-30分钟")
    elif choice == '3':
        limit = None
        max_workers = 10  # 进一步减少并发数，因为每个样本调用更多
        print("⚠️  全量评估：213个样本，预计需要1.5-3小时！")
        confirm = input("确认要进行全量评估吗？(y/N): ").strip().lower()
        if confirm != 'y':
            print("已取消评估")
            exit(0)
    else:
        print("无效选择，使用测试模式")
        limit = 10
        max_workers = 2
    
    # 执行评估
    results = evaluate_dataset(input_file, output_file, max_workers, limit, disease_list_file, use_context)
    
    # 简单分析
    print("\n=== 简单准确率分析 ===")
    analysis = simple_accuracy_analysis(results)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print(f"\n结果已保存到: {output_file}")
