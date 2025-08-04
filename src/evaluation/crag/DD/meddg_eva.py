import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import re

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from crag import corrective_rag_pipeline

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载MEDDG数据集并按dialog分组
    
    Args:
        file_path: 数据集文件路径
    
    Returns:
        按dialog分组的数据列表
    """
    dialogs = []
    current_dialog = []
    current_dialog_id = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 检查是否是dialog分隔符
            if line.startswith('dialog'):
                # 如果有当前dialog且不为空，保存它
                if current_dialog_id is not None and current_dialog:
                    dialog_data = {
                        'id': current_dialog_id,
                        'dialog_lines': current_dialog,
                        'line_start': current_dialog[0]['line_num'] if current_dialog else line_num,
                        'line_end': current_dialog[-1]['line_num'] if current_dialog else line_num
                    }
                    dialogs.append(dialog_data)
                
                # 开始新的dialog
                current_dialog_id = len(dialogs)  # 使用索引作为ID
                current_dialog = []
            else:
                # 尝试解析JSON行
                try:
                    json_data = json.loads(line)
                    json_data['line_num'] = line_num
                    current_dialog.append(json_data)
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON格式错误，跳过: {line[:50]}...")
                    continue
    
    # 处理最后一个dialog
    if current_dialog_id is not None and current_dialog:
        dialog_data = {
            'id': current_dialog_id,
            'dialog_lines': current_dialog,
            'line_start': current_dialog[0]['line_num'] if current_dialog else 0,
            'line_end': current_dialog[-1]['line_num'] if current_dialog else 0
        }
        dialogs.append(dialog_data)
    
    print(f"加载完成：共{len(dialogs)}个对话，总计{line_num}行")
    return dialogs

def preprocess_dialog(dialog_lines: List[Dict[str, Any]]) -> str:
    """
    将dialog中的Patient和Doctor对话拼接成字符串
    
    Args:
        dialog_lines: 对话行列表
    
    Returns:
        拼接后的对话字符串
    """
    dialog_parts = []
    
    for line in dialog_lines:
        speaker_id = line.get('id', '')
        sentence = line.get('Sentence', '')
        
        # 只处理Patient和Doctor的对话
        if speaker_id in ['Patients', 'Doctor'] and sentence.strip():
            # 统一Patient和Patients的命名
            speaker = 'Patient' if speaker_id == 'Patients' else speaker_id
            dialog_parts.append(f"{speaker}: {sentence}")
    
    return '\n'.join(dialog_parts)

def extract_diseases_from_diagnosis(diagnosis_text: str) -> List[str]:
    """
    从诊断文本中提取疾病名称
    根据1.md中的格式：<final_diagnosis>{"diseases": [产气杆菌肠炎]}</final_diagnosis>
    
    Args:
        diagnosis_text: 诊断文本
    
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
            
def process_single_item(item: Dict[str, Any], disease_list_file: str = None) -> Dict[str, Any]:
    """
    处理单个对话数据项
    
    Args:
        item: 包含对话数据的字典
        disease_list_file: 疾病列表文件路径，可选
    
    Returns:
        处理结果字典
    """
    try:
        # 预处理对话：拼接Patient和Doctor的对话
        input_dialog = preprocess_dialog(item['dialog_lines'])
        
        # 检查对话是否为空
        if not input_dialog.strip():
            return {
                'id': item['id'],
                'input_dialog': '',
                'raw_diagnosis': "对话内容为空，无法诊断",
                'predicted_diseases': ["无对话内容"],
                'processing_time': 0,
                'status': 'empty_dialog'
            }

        # 调用LLM Medication RAG诊断流程，使用静默模式减少日志输出
        start_time = time.time()
        diagnosis_result = corrective_rag_pipeline(input_dialog, disease_list_file=disease_list_file)
        end_time = time.time()

        # 提取疾病信息
        predicted_diseases = extract_diseases_from_diagnosis(diagnosis_result)
        
        result = {
            'id': item['id'],
            'input_dialog': input_dialog,
            'raw_diagnosis': diagnosis_result,
            'predicted_diseases': predicted_diseases,
            'processing_time': round(end_time - start_time, 2),
            'status': 'success',
            'dialog_lines_count': len(item['dialog_lines']),
            'line_range': f"{item['line_start']}-{item['line_end']}"
        }
        
        print(f"✓ 完成ID {item['id']}: {len(input_dialog)}字符 -> {predicted_diseases}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} 处理失败: {str(e)}")
        return {
            'id': item['id'],
            'input_dialog': preprocess_dialog(item.get('dialog_lines', [])),
            'raw_diagnosis': f"处理错误: {str(e)}",
            'predicted_diseases': ["处理失败"],
            'processing_time': 0,
            'status': 'error',
            'dialog_lines_count': len(item.get('dialog_lines', [])),
            'line_range': f"{item.get('line_start', 0)}-{item.get('line_end', 0)}"
        }

def evaluate_dataset(input_file: str, output_file: str, max_workers: int = 50, limit: int = None, disease_list_file: str = None):
    """
    评估整个MEDDG数据集
    
    Args:
        input_file: 输入数据集文件路径
        output_file: 输出结果文件路径
        max_workers: 并发线程数
        limit: 限制处理的数据条数，None表示处理全部
        disease_list_file: 疾病列表文件路径，可选
    """
    print(f"开始评估MEDDG数据集 (LLM Medication RAG): {input_file}")
    print(f"并发线程数: {max_workers}")
    
    if disease_list_file:
        print(f"使用疾病列表约束: {disease_list_file}")
    else:
        print("不使用疾病列表约束")
    
    # 加载数据集
    print("加载MEDDG数据集...")
    dataset = load_dataset(input_file)
    
    if limit:
        dataset = dataset[:limit]
        print(f"限制处理前 {limit} 个对话")
    
    print(f"总共 {len(dataset)} 个对话")
    
    # 并发处理
    print("\n开始并发处理 (LLM Medication RAG)...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，传入疾病列表文件
        future_to_item = {executor.submit(process_single_item, item, disease_list_file): item for item in dataset}
        
        # 收集结果
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
    
    # 按id排序确保顺序正确
    results.sort(key=lambda x: x['id'])
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    print(f"\n处理完成! 总耗时: {total_time}秒")
    print(f"平均每个对话耗时: {round(total_time/len(dataset), 2)}秒")
    
    # 统计成功失败数
    success_count = sum(1 for r in results if r['status'] == 'success')
    empty_dialog_count = sum(1 for r in results if r['status'] == 'empty_dialog')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"成功: {success_count}, 空对话: {empty_dialog_count}, 失败: {error_count}")
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("MEDDG LLM Medication RAG评估完成!")
    return results

def simple_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    简单的统计分析（无准确率，因为没有ground_truth）
    
    Args:
        results: 评估结果列表
    
    Returns:
        分析统计信息
    """
    total = len(results)
    success_results = [r for r in results if r['status'] == 'success']
    success_count = len(success_results)
    empty_dialog_count = sum(1 for r in results if r['status'] == 'empty_dialog')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    # 统计平均对话长度
    avg_dialog_length = 0
    avg_lines_count = 0
    if success_results:
        total_chars = sum(len(r['input_dialog']) for r in success_results)
        total_lines = sum(r['dialog_lines_count'] for r in success_results)
        avg_dialog_length = round(total_chars / len(success_results), 2)
        avg_lines_count = round(total_lines / len(success_results), 2)
    
    return {
        'total': total,
        'success_count': success_count,
        'empty_dialog_count': empty_dialog_count,
        'error_count': error_count,
        'avg_dialog_length_chars': avg_dialog_length,
        'avg_dialog_lines_count': avg_lines_count,
        'success_rate': round(success_count / total, 4) if total > 0 else 0.0
    }

if __name__ == "__main__":
    # 配置文件路径
    input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/MEDDG/test.txt"
    output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/final_result/meddg/crag"
    output_file = os.path.join(output_dir, "meddg_evaluation_results_top5.jsonl")
    
    # 疾病列表文件路径配置（可选）
    # 设置为 None 表示不使用疾病列表约束
    disease_list_file = None  # MEDDG数据集不使用疾病列表约束
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    print("=== MEDDG对话数据集评估 (LLM Medication RAG) ===")
    choice = input("选择模式:\n1. 测试模式(前10个对话)\n2. 小批量(前50个对话)\n3. 全量评估\n请选择(1/2/3): ").strip()
    
    if choice == '1':
        limit = 10
        max_workers = 3
    elif choice == '2':
        limit = 50
        max_workers = 20
    elif choice == '3':
        limit = None
        max_workers = 50
    else:
        print("无效选择，使用测试模式")
        limit = 10
        max_workers = 3
    
    # 执行评估
    results = evaluate_dataset(input_file, output_file, max_workers, limit, disease_list_file)
    
    # 简单分析
    print("\n=== 简单统计分析 (LLM Medication RAG) ===")
    analysis = simple_analysis(results)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print(f"\n结果已保存到: {output_file}")
    print("注意：MEDDG数据集无ground_truth，因此无法计算准确率。")
    print("如需质量评估，请运行对应的meddg_prediction.py文件。")