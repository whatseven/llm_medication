import json
import os
import re
from openai import OpenAI
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import (
    TEXTGRAD_COMPREHENSIVE_CRITIC_PROMPT,
    TEXTGRAD_PROMPT_OPTIMIZER_PROMPT,
    TEXTGRAD_DIAGNOSIS_GENERATOR_PROMPT
)


class LLMMedTextGrad:
    """
    简化的医疗诊断TextGrad优化器
    3步流程：综合评估 → 提示词优化 → 独立诊断生成
    """
    
    def __init__(self, model_name: str = "deepseek", num_iterations: int = 1, verbose: bool = False):
        """
        初始化TextGrad优化器
        
        Args:
            model_name: 使用的模型名称
            num_iterations: 迭代次数，默认1次
            verbose: 是否输出详细日志
        """
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.verbose = verbose
        
        # 获取模型配置
        self.model_config = MODELS[model_name]
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"]
        )
    
    def _log(self, message: str):
        """记录日志"""
        if self.verbose:
            print(f"[TextGrad] {message}")
    
    def _call_llm(self, prompt: str, task_description: str) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            task_description: 任务描述（用于日志）
            
        Returns:
            LLM响应内容
        """
        self._log(f"调用LLM - {task_description}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=[
                    {"role": "system", "content": "你是一位专业的医疗助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            self._log(f"LLM调用失败 - {task_description}: {str(e)}")
            return f"[LLM调用失败: {str(e)}]"
    
    def _prepare_comprehensive_context(self, user_input: str, disease_information: list, disease_list: str) -> str:
        """准备完整的上下文信息"""
        context = f"患者完整病情描述：\n{user_input}\n\n"
        
        # 候选疾病信息（向量库+图数据库融合）
        if disease_information:
            context += "相关疾病知识库信息：\n"
            for i, disease in enumerate(disease_information, 1):
                context += f"{i}. 疾病名称：{disease.get('name', '未知疾病')}\n"
                context += f"   基础描述：{disease.get('desc', '无描述')}\n"
                context += f"   相关症状：{disease.get('symptom', '无症状信息')}\n"
                context += f"   匹配相似度：{disease.get('similarity_score', 0):.3f}\n"
                
                # 图数据库增强信息
                has_graph_info = False
                if 'graph_cause' in disease:
                    context += f"   详细病因：{disease['graph_cause']}\n"
                    has_graph_info = True
                if 'department' in disease:
                    context += f"   推荐科室：{disease['department']}\n"
                    has_graph_info = True
                if 'complications' in disease:
                    context += f"   可能并发症：{disease['complications']}\n"
                    has_graph_info = True
                
                if not has_graph_info:
                    context += f"   (仅基础信息，无详细医学资料)\n"
                
                context += "\n"
        else:
            context += "相关疾病知识库信息：无可用疾病信息\n\n"
        
        # 疾病列表约束
        context += f"诊断约束条件：\n"
        if disease_list and disease_list != "无约束":
            context += f"必须从以下疾病列表中选择：{disease_list}\n"
        else:
            context += f"无疾病列表限制，可诊断任何合适疾病\n"
        
        return context
    
    def _load_disease_list(self, disease_list_file: str = None) -> str:
        """加载疾病列表"""
        if not disease_list_file or not os.path.exists(disease_list_file):
            return "无约束"
        
        try:
            with open(disease_list_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return "无约束"
                
                # 尝试解析为列表格式
                try:
                    import ast
                    disease_list = ast.literal_eval(content)
                    if isinstance(disease_list, list):
                        return ', '.join(disease_list)
                except:
                    # 按行读取
                    lines = content.split('\n')
                    diseases = [line.strip() for line in lines if line.strip()]
                    if diseases:
                        return ', '.join(diseases)
                        
            return "无约束"
        except Exception as e:
            self._log(f"读取疾病列表失败: {str(e)}")
            return "无约束"
    
    def _extract_diagnosis_text(self, diagnosis_result) -> str:
        """从诊断结果中提取诊断文本"""
        if not diagnosis_result:
            return ""
        
        diagnosis_str = str(diagnosis_result)
        
        try:
            # 尝试从<final_diagnosis>标签中提取
            pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
            match = re.search(pattern, diagnosis_str, re.DOTALL)
            
            if match:
                diagnosis_data = json.loads(match.group(1))
                diseases = diagnosis_data.get('diseases', [])
                if isinstance(diseases, list) and diseases:
                    return diseases[0]  # 返回第一个疾病
                elif isinstance(diseases, str):
                    return diseases
            
            # 备选提取模式
            for pattern in [r'诊断[：:]\s*([^。\n]+)', r'考虑[：:]?\s*([^。\n，,]+)']:
                matches = re.findall(pattern, diagnosis_str)
                if matches:
                    return matches[0].strip()
            
            return diagnosis_str[:100].strip()
            
        except Exception as e:
            return diagnosis_str[:100].strip()
    
    def _ensure_diagnosis_format(self, diagnosis_text: str) -> str:
        """确保诊断格式符合标准"""
        if not diagnosis_text or not diagnosis_text.strip():
            return '<final_diagnosis>{"diseases": ["未知疾病"]}</final_diagnosis>'
        
        # 如果已经有格式，直接返回
        if '<final_diagnosis>' in diagnosis_text:
            return diagnosis_text
        
        # 清理诊断文本
        clean_text = diagnosis_text.strip()
        clean_text = clean_text.strip('"\'""''()[]{}（）【】')
        
        # 如果包含多个疾病，取第一个
        if '、' in clean_text:
            clean_text = clean_text.split('、')[0]
        elif '，' in clean_text:
            clean_text = clean_text.split('，')[0]
        elif ',' in clean_text:
            clean_text = clean_text.split(',')[0]
        
        clean_text = clean_text.strip()
        
        # 包装成标准格式
        return f'<final_diagnosis>{{"diseases": ["{clean_text}"]}}</final_diagnosis>'
    
    def optimize_diagnosis(self, user_input: str, disease_information: list, 
                          initial_diagnosis, disease_list_file: str = None):
        """
        主要优化接口 - 3步简化流程
        
        Args:
            user_input: 患者查询
            disease_information: 融合的疾病信息列表
            initial_diagnosis: 初步诊断结果
            disease_list_file: 疾病列表文件路径
            
        Returns:
            dict: {"is_correct": bool, "optimized_diagnosis": str} 或 {"is_correct": bool}
        """
        self._log("开始TextGrad诊断优化")
        
        # 数据验证
        if not user_input or not user_input.strip():
            self._log("错误：用户输入为空")
            return {"is_correct": True}
        
        if not disease_information or len(disease_information) == 0:
            self._log("警告：无疾病信息可用，跳过优化")
            return {"is_correct": True}
        
        # 检查disease_information数据质量
        valid_diseases = 0
        graph_enhanced_diseases = 0
        for disease in disease_information:
            if disease.get('name') and disease.get('desc'):
                valid_diseases += 1
                if 'graph_cause' in disease or 'department' in disease or 'complications' in disease:
                    graph_enhanced_diseases += 1
        
        self._log(f"数据质量检查：{valid_diseases}个有效疾病，{graph_enhanced_diseases}个包含图数据库增强信息")
        
        if valid_diseases == 0:
            self._log("错误：无有效疾病信息，跳过优化")
            return {"is_correct": True}
        
        try:
            # 准备数据
            current_diagnosis = self._extract_diagnosis_text(initial_diagnosis)
            disease_list = self._load_disease_list(disease_list_file)
            comprehensive_context = self._prepare_comprehensive_context(user_input, disease_information, disease_list)
            
            self._log(f"初始诊断: {current_diagnosis}")
            
            best_diagnosis = current_diagnosis
            
            # 进行指定次数的迭代优化
            for iteration in range(self.num_iterations):
                self._log(f"开始第 {iteration + 1}/{self.num_iterations} 轮迭代")
                
                # 步骤1: 综合评估与改进建议
                evaluation_prompt = TEXTGRAD_COMPREHENSIVE_CRITIC_PROMPT.format(
                    comprehensive_context=comprehensive_context,
                    current_diagnosis=current_diagnosis
                )
                improvement_suggestions = self._call_llm(evaluation_prompt, "综合评估")
                
                if "[LLM调用失败" in improvement_suggestions:
                    self._log(f"第 {iteration + 1} 轮评估失败")
                    break
                
                # 步骤2: 提示词优化
                optimizer_prompt = TEXTGRAD_PROMPT_OPTIMIZER_PROMPT.format(
                    comprehensive_context=comprehensive_context,
                    current_diagnosis=current_diagnosis,
                    improvement_suggestions=improvement_suggestions
                )
                optimized_prompt = self._call_llm(optimizer_prompt, "提示词优化")
                
                if "[LLM调用失败" in optimized_prompt:
                    self._log(f"第 {iteration + 1} 轮提示词优化失败")
                    break
                
                # 步骤3: 独立诊断生成
                diagnosis_prompt = TEXTGRAD_DIAGNOSIS_GENERATOR_PROMPT.format(
                    current_diagnosis=current_diagnosis,
                    optimized_prompt=optimized_prompt,
                    comprehensive_context=comprehensive_context
                )
                new_diagnosis = self._call_llm(diagnosis_prompt, "独立诊断生成")
                
                if "[LLM调用失败" not in new_diagnosis:
                    new_diagnosis_formatted = self._ensure_diagnosis_format(new_diagnosis)
                    new_diagnosis_text = self._extract_diagnosis_text(new_diagnosis_formatted)
                    
                    if new_diagnosis_text and new_diagnosis_text != current_diagnosis:
                        current_diagnosis = new_diagnosis_text
                        best_diagnosis = new_diagnosis_formatted
                        self._log(f"第 {iteration + 1} 轮优化: {new_diagnosis_text}")
                    else:
                        self._log(f"第 {iteration + 1} 轮无显著改进")
                else:
                    self._log(f"第 {iteration + 1} 轮诊断生成失败")
            
            # 判断是否有改进
            initial_diagnosis_text = self._extract_diagnosis_text(initial_diagnosis)
            if best_diagnosis != initial_diagnosis and current_diagnosis != initial_diagnosis_text:
                self._log(f"优化完成，从 '{initial_diagnosis_text}' 优化为 '{current_diagnosis}'")
                return {
                    "is_correct": False, 
                    "optimized_diagnosis": best_diagnosis
                }
            else:
                self._log("优化完成，诊断无显著改进")
                return {"is_correct": True}
                
        except Exception as e:
            self._log(f"TextGrad优化过程出错: {str(e)}")
            return {"is_correct": True}  # 出错时返回原诊断 