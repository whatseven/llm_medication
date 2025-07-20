# RJUA数据集评估脚本使用说明

## 📁 文件说明

### 1. `rjua_cn_eva.py` - 基础诊断评估
进行基础的医疗诊断评估，输出诊断结果和简单准确率。

### 2. `rjua_cn_prediction.py` - LLM质量评估  
基于第一步的结果，使用LLM进行深度质量评估。

## 🔧 配置方式

### 第一步：修改 `rjua_cn_eva.py`

在文件顶部的配置区域修改参数：

```python
# ==================== 配置参数区域 ====================
# 输入数据集文件路径
input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/RJUA_CN/RJUA_test.json"

# 输出目录和文件名
output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/RJUACN"
output_file = os.path.join(output_dir, "rjua_evaluation_results1.jsonl")

# 疾病列表文件路径配置（可选）
disease_list_file = None  # 不使用疾病列表约束
# disease_list_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/RJUA_CN/disease.txt"  # 使用疾病列表约束

# 输入模式配置
use_context = False  # False: 仅使用问题, True: 使用问题+知识背景
# ====================================================
```

### 第二步：修改 `rjua_cn_prediction.py`

在文件顶部的配置区域修改参数：

```python
# ==================== 配置参数区域 ====================
# 输入评估结果文件路径（第一步的输出文件）
input_file = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/RJUACN/rjua_evaluation_results1.jsonl"

# 输出目录和文件名
output_dir = "/home/ubuntu/ZJQ/llm_medication/llm_medication/src/data/result/RJUACN"
output_file = os.path.join(output_dir, "rjua_quality_evaluation_results1.jsonl")

# 评估模型配置
model_name = DEFAULT_MODEL  # 使用默认模型
# model_name = "qwen2.5:72b"  # 或指定其他模型
# ====================================================
```

## 🚀 使用步骤

### 步骤1：运行基础评估
```bash
cd /home/ubuntu/ZJQ/llm_medication/llm_medication/src/evaluation/llm_medication_rag/DD
python3 rjua_cn_eva.py
```

选择运行模式：
- 1. 测试模式(前10条)
- 2. 小批量(前50条) 
- 3. 全量评估

### 步骤2：运行质量评估
```bash
python3 rjua_cn_prediction.py
```

选择评估模式：
- 1. 测试模式(前5条)
- 2. 小批量(前20条)
- 3. 中批量(前50条)
- 4. 全量评估

## 📝 配置示例

### 示例1：仅问题模式，无疾病列表约束
```python
# rjua_cn_eva.py 配置
input_file = "/path/to/RJUA_test.json"
output_file = os.path.join(output_dir, "rjua_question_only_no_list.jsonl")
disease_list_file = None
use_context = False

# rjua_cn_prediction.py 配置
input_file = "/path/to/rjua_question_only_no_list.jsonl"
output_file = os.path.join(output_dir, "quality_rjua_question_only_no_list.jsonl")
model_name = DEFAULT_MODEL
```

### 示例2：问题+背景模式，使用疾病列表约束
```python
# rjua_cn_eva.py 配置
input_file = "/path/to/RJUA_test.json"
output_file = os.path.join(output_dir, "rjua_context_with_list.jsonl")
disease_list_file = "/path/to/disease.txt"
use_context = True

# rjua_cn_prediction.py 配置
input_file = "/path/to/rjua_context_with_list.jsonl"
output_file = os.path.join(output_dir, "quality_rjua_context_with_list.jsonl")
model_name = "qwen2.5:72b"
```

## 📊 输出文件格式

### 基础评估输出 (`rjua_evaluation_results*.jsonl`)
```json
{
  "id": "1",
  "ground_truth_disease": ["睾丸炎", "睾丸扭转", "脓毒血症"],
  "ground_truth_answer": "您好，根据您的症状描述...",
  "ground_truth_advice": "阴囊探查术、留取血和尿培养...",
  "input_text": "医生您好，我昨天左边的睾丸痛...",
  "raw_diagnosis": "根据患者症状...<final_diagnosis>...</final_diagnosis>",
  "predicted_diseases": ["睾丸炎", "附睾炎"],
  "processing_time": 12.34,
  "status": "success",
  "use_context": false
}
```

### 质量评估输出 (`rjua_quality_evaluation_results*.jsonl`)
```json
{
  "id": "1",
  "original_ground_truth": ["睾丸炎", "睾丸扭转", "脓毒血症"],
  "model_prediction": ["睾丸炎", "附睾炎"],
  "input_text": "医生您好，我昨天左边的睾丸痛...",
  "raw_diagnosis": "根据患者症状...",
  "use_context": false,
  "llm_evaluation_result": 1,
  "llm_evaluation_reasoning": "评估分析过程...<r>1</r>",
  "evaluation_time": 3.45,
  "status": "success"
}
```

## 🎯 快速切换配置

### 常用配置模板

#### 配置A：测试RJUA基础功能
```python
# rjua_cn_eva.py
output_file = os.path.join(output_dir, "test_basic.jsonl")
disease_list_file = None
use_context = False

# rjua_cn_prediction.py  
input_file = "/path/to/test_basic.jsonl"
output_file = os.path.join(output_dir, "quality_test_basic.jsonl")
```

#### 配置B：对比有无背景知识
```python
# 实验1：仅问题
output_file = os.path.join(output_dir, "compare_question_only.jsonl")
use_context = False

# 实验2：问题+背景
output_file = os.path.join(output_dir, "compare_with_context.jsonl") 
use_context = True
```

#### 配置C：对比有无疾病列表约束
```python
# 实验1：无约束
disease_list_file = None
output_file = os.path.join(output_dir, "compare_no_constraint.jsonl")

# 实验2：有约束
disease_list_file = "/path/to/disease.txt"
output_file = os.path.join(output_dir, "compare_with_constraint.jsonl")
```

## ⚠️ 注意事项

1. **文件路径**：确保输入文件路径正确存在
2. **文件名冲突**：不同实验使用不同的输出文件名
3. **依赖关系**：第二步的输入文件必须是第一步的输出文件
4. **模型配置**：确保指定的模型在 `MODELS` 配置中存在
5. **并发控制**：根据系统性能调整并发数量 