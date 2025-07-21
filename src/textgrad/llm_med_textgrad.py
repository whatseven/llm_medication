import json
import os
import re
from openai import OpenAI
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import (
    TEXTGRAD_KNOWLEDGE_CRITIC_PROMPT,
    TEXTGRAD_PATIENT_CRITIC_PROMPT,
    TEXTGRAD_DIAGNOSIS_GRADIENT_PROMPT,
    TEXTGRAD_PROMPT_OPTIMIZER_PROMPT,
    TEXTGRAD_INITIAL_PROMPT_TEMPLATE
)
from src.model.doctor import diagnose


class LLMMedTextGrad:
    """
    医疗诊断TextGrad优化器
    基于Med-TextGrad实现，用于迭代优化医疗诊断结果
    """
    
    def __init__(self, model_name: str = "deepseek", num_iterations: int = 3, verbose: bool = False):
        """
        初始化TextGrad优化器
        
        Args:
            model_name: 使用的模型名称
            num_iterations: 迭代次数
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
    
    def _call_llm(self, task_description: str, prompt: str, max_tokens: int = 4096) -> str:
        """
        调用LLM API
        
        Args:
            task_description: 任务描述（用于日志）
            prompt: 提示词
            max_tokens: 最大tokens数量
            
        Returns:
            LLM响应内容
        """
        self._log(f"调用LLM - {task_description}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=[
                    {"role": "system", "content": "你是一位专业的医疗助手，专门负责医疗诊断优化和评估。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            
            content = response.choices[0].message.content
            self._log(f"LLM响应 - {task_description}: {content[:100]}...")
            return content if content else ""
            
        except Exception as e:
            self._log(f"LLM调用失败 - {task_description}: {str(e)}")
            return f"[LLM调用失败: {str(e)}]"
    
    def _extract_diagnosis_text(self, diagnosis_result) -> str:
        """
        从诊断结果中提取诊断文本
        
        Args:
            diagnosis_result: 诊断结果（通常是包含<final_diagnosis>标签的字符串）
            
        Returns:
            诊断文本字符串
        """
        if not diagnosis_result:
            return ""
        
        diagnosis_str = str(diagnosis_result)
        
        try:
            # 尝试从<final_diagnosis>标签中提取
            import re
            import json
            
            pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
            match = re.search(pattern, diagnosis_str, re.DOTALL)
            
            if match:
                diagnosis_data = json.loads(match.group(1))
                diseases = diagnosis_data.get('diseases', [])
                if isinstance(diseases, list) and diseases:
                    return diseases[0]  # 返回第一个疾病
                elif isinstance(diseases, str):
                    return diseases
            
            # 如果没有找到标签，尝试其他提取模式
            for pattern in [
                r'诊断[：:]\s*([^。\n]+)', 
                r'可能的疾病[：:]\s*([^。\n]+)',
                r'初步诊断[：:]\s*([^。\n]+)', 
                r'考虑[：:]?\s*([^。\n，,]+)'
            ]:
                matches = re.findall(pattern, diagnosis_str)
                if matches:
                    return matches[0].strip()
            
            # 如果都没有匹配，返回原始字符串的前100个字符
            return diagnosis_str[:100].strip()
            
        except Exception as e:
            # 出错时返回原始字符串
            return diagnosis_str[:100].strip()
    
    def _load_disease_list(self, disease_list_file: str = None) -> str:
        """
        加载疾病列表
        
        Args:
            disease_list_file: 疾病列表文件路径
            
        Returns:
            格式化的疾病列表字符串
        """
        if not disease_list_file or not os.path.exists(disease_list_file):
            return "无约束"
        
        try:
            with open(disease_list_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return "无约束"
                
                # 尝试解析为Python列表格式
                try:
                    import ast
                    disease_list = ast.literal_eval(content)
                    if isinstance(disease_list, list):
                        return ', '.join(disease_list)
                except:
                    # 如果不是列表格式，按行读取
                    lines = content.split('\n')
                    diseases = [line.strip() for line in lines if line.strip()]
                    if diseases:
                        return ', '.join(diseases)
                        
            return "无约束"
        except Exception as e:
            self._log(f"读取疾病列表失败: {str(e)}")
            return "无约束"
    
    def _format_disease_information(self, disease_information: list) -> str:
        """
        格式化疾病信息为上下文字符串
        
        Args:
            disease_information: 疾病信息列表
            
        Returns:
            格式化的医学知识上下文
        """
        if not disease_information:
            return "无可用医学知识"
        
        context = "医学知识上下文：\n"
        for i, disease in enumerate(disease_information, 1):
            context += f"{i}. 疾病：{disease.get('name', '未知')}\n"
            context += f"   描述：{disease.get('desc', '无描述')}\n"
            context += f"   症状：{disease.get('symptom', '无症状信息')}\n"
            context += f"   相似度：{disease.get('similarity_score', 0):.4f}\n"
            
            # 添加图数据库信息（如果存在）
            if 'graph_cause' in disease:
                context += f"   病因：{disease['graph_cause']}\n"
            if 'department' in disease:
                context += f"   科室：{disease['department']}\n"
            if 'complications' in disease:
                context += f"   并发症：{disease['complications']}\n"
            context += "\n"
        
        return context
    
    def knowledge_criterion_agent(self, diagnosis: str, medical_context: str, disease_list: str) -> str:
        """
        医学知识批判代理
        
        Args:
            diagnosis: 当前诊断
            medical_context: 医学知识上下文
            disease_list: 疾病列表
            
        Returns:
            批判意见
        """
        prompt = TEXTGRAD_KNOWLEDGE_CRITIC_PROMPT.format(
            medical_context=medical_context,
            current_diagnosis=diagnosis,
            disease_list=disease_list
        )
        
        return self._call_llm("医学知识批判", prompt)
    
    def patient_criterion_agent(self, diagnosis: str, patient_query: str) -> str:
        """
        患者查询批判代理
        
        Args:
            diagnosis: 当前诊断
            patient_query: 患者查询
            
        Returns:
            批判意见
        """
        prompt = TEXTGRAD_PATIENT_CRITIC_PROMPT.format(
            patient_query=patient_query,
            current_diagnosis=diagnosis
        )
        
        return self._call_llm("患者查询批判", prompt)
    
    def compute_diagnosis_gradient(self, diagnosis: str, knowledge_critique: str, 
                                 patient_critique: str, disease_list: str) -> str:
        """
        计算诊断文本梯度
        
        Args:
            diagnosis: 当前诊断
            knowledge_critique: 医学知识批判
            patient_critique: 患者查询批判
            disease_list: 疾病列表
            
        Returns:
            诊断改进指导
        """
        prompt = TEXTGRAD_DIAGNOSIS_GRADIENT_PROMPT.format(
            current_diagnosis=diagnosis,
            knowledge_critique=knowledge_critique,
            patient_critique=patient_critique,
            disease_list=disease_list
        )
        
        return self._call_llm("计算诊断梯度", prompt)
    
    def optimize_prompt(self, original_prompt: str, diagnosis: str, 
                       diagnosis_gradient: str, patient_query: str) -> str:
        """
        优化提示词
        
        Args:
            original_prompt: 原始提示词
            diagnosis: 当前诊断
            diagnosis_gradient: 诊断改进指导
            patient_query: 患者查询
            
        Returns:
            优化后的提示词
        """
        prompt = TEXTGRAD_PROMPT_OPTIMIZER_PROMPT.format(
            original_prompt=original_prompt,
            current_diagnosis=diagnosis,
            diagnosis_gradient=diagnosis_gradient,
            patient_query=patient_query
        )
        
        return self._call_llm("优化提示词", prompt)
    
    def generator_agent(self, optimized_prompt: str, current_diagnosis: str, 
                       user_input: str, disease_information: list, disease_list_file: str = None) -> str:
        """
        生成器代理 - 使用优化后的提示词生成改进的诊断
        
        Args:
            optimized_prompt: 优化后的提示词
            current_diagnosis: 当前诊断
            user_input: 用户输入
            disease_information: 疾病信息
            disease_list_file: 疾病列表文件
            
        Returns:
            优化后的诊断
        """
        # 构建完整的生成提示词
        full_prompt = f"{optimized_prompt}\n\n当前诊断：{current_diagnosis}\n\n请提供优化后的诊断，只输出疾病名称："
        
        try:
            # 调用LLM生成优化诊断
            optimized_diagnosis = self._call_llm("生成优化诊断", full_prompt)
            
            # 清理输出，提取疾病名称
            cleaned_diagnosis = self._clean_diagnosis_output(optimized_diagnosis)
            
            # 简单验证和清理
            if "[LLM调用失败" in cleaned_diagnosis:
                return current_diagnosis
            
            return cleaned_diagnosis
            
        except Exception as e:
            self._log(f"生成器代理失败: {str(e)}")
            return current_diagnosis
    
    def _clean_diagnosis_output(self, raw_output: str) -> str:
        """
        清理LLM输出，确保格式一致
        
        Args:
            raw_output: LLM原始输出
            
        Returns:
            清理后的诊断字符串
        """
        if not raw_output or not isinstance(raw_output, str):
            return ""
        
        # 基本清理
        cleaned = raw_output.strip()
        
        # 移除常见的多余文本
        unwanted_prefixes = [
            "优化后的诊断：", "诊断：", "疾病：", "最终诊断：", 
            "建议诊断：", "推荐诊断：", "诊断结果：", "答案：",
            "优化诊断：", "改进诊断："
        ]
        
        for prefix in unwanted_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # 移除引号和其他标点
        cleaned = cleaned.strip('"\'""''()[]{}（）【】')
        
        # 移除换行符和多余空格
        cleaned = ' '.join(cleaned.split())
        
        # 如果包含多个疾病，取第一个
        if '、' in cleaned:
            cleaned = cleaned.split('、')[0]
        elif '，' in cleaned:
            cleaned = cleaned.split('，')[0]
        elif ',' in cleaned:
            cleaned = cleaned.split(',')[0]
        
        return cleaned.strip()
    
    def optimize_diagnosis(self, user_input: str, disease_information: list, 
                          initial_diagnosis, disease_list_file: str = None):
        """
        主要优化接口 - 即插即用替换iteration模块
        
        Args:
            user_input: 患者查询
            disease_information: 融合的疾病信息列表
            initial_diagnosis: 初步诊断结果
            disease_list_file: 疾病列表文件路径
            
        Returns:
            dict: {"is_correct": bool, "optimized_diagnosis": str} 或 {"is_correct": bool}
        """
        self._log("开始TextGrad诊断优化")
        
        try:
            # 提取初始诊断文本
            current_diagnosis = self._extract_diagnosis_text(initial_diagnosis)
            self._log(f"初始诊断: {current_diagnosis}")
            
            # 加载疾病列表和格式化医学上下文
            disease_list = self._load_disease_list(disease_list_file)
            medical_context = self._format_disease_information(disease_information)
            
            # 构建初始提示词
            current_prompt = TEXTGRAD_INITIAL_PROMPT_TEMPLATE.format(
                patient_query=user_input,
                medical_context=medical_context,
                disease_list=disease_list
            )
            
            # 固定3轮迭代优化
            best_diagnosis = current_diagnosis
            
            for iteration in range(self.num_iterations):
                self._log(f"开始第 {iteration + 1}/{self.num_iterations} 轮迭代")
                
                # 1. 生成优化诊断
                refined_diagnosis = self.generator_agent(
                    current_prompt, current_diagnosis, user_input, disease_information, disease_list_file
                )
                
                if "[LLM调用失败" in refined_diagnosis:
                    self._log(f"第 {iteration + 1} 轮生成失败，使用前一轮结果")
                    break
                
                current_diagnosis = refined_diagnosis
                self._log(f"第 {iteration + 1} 轮优化诊断: {current_diagnosis}")
                
                # 2. 医学知识批判
                knowledge_critique = self.knowledge_criterion_agent(
                    current_diagnosis, medical_context, disease_list
                )
                
                # 3. 患者查询批判
                patient_critique = self.patient_criterion_agent(
                    current_diagnosis, user_input
                )
                
                # 4. 计算诊断改进梯度
                diagnosis_gradient = self.compute_diagnosis_gradient(
                    current_diagnosis, knowledge_critique, patient_critique, disease_list
                )
                
                # 5. 优化提示词（用于下一轮）
                if iteration < self.num_iterations - 1:
                    current_prompt = self.optimize_prompt(
                        current_prompt, current_diagnosis, diagnosis_gradient, user_input
                    )
                    
                    if "[LLM调用失败" in current_prompt:
                        self._log(f"第 {iteration + 1} 轮提示词优化失败，使用原提示词")
                        # 恢复到初始提示词
                        current_prompt = TEXTGRAD_INITIAL_PROMPT_TEMPLATE.format(
                            patient_query=user_input,
                            medical_context=medical_context,
                            disease_list=disease_list
                        )
                
                # 更新最佳诊断
                best_diagnosis = current_diagnosis
            
            # 判断是否有显著改进
            initial_diagnosis_text = self._extract_diagnosis_text(initial_diagnosis)
            if best_diagnosis != initial_diagnosis_text and best_diagnosis.strip():
                self._log(f"优化完成，从 '{initial_diagnosis_text}' 优化为 '{best_diagnosis}'")
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


def iterative_diagnose_textgrad(symptoms, vector_results, graph_data, doctor_diagnosis, disease_list_file=None):
    """
    TextGrad版本的迭代诊断函数 - 兼容原iteration接口
    
    Args:
        symptoms: 患者症状描述
        vector_results: 候选疾病信息（已废弃，改用disease_information）
        graph_data: 图数据库信息（已废弃，现已融合）
        doctor_diagnosis: 初步诊断结果
        disease_list_file: 疾病列表文件路径
        
    Returns:
        dict: 与原iteration模块相同的返回格式
    """
    # 注意：这个函数主要用于兼容，实际应该在main.py中直接调用optimize_diagnosis
    # 这里简化处理，假设disease_information已经融合了所有信息
    
    optimizer = LLMMedTextGrad(verbose=False)
    
    # 由于接口限制，这里暂时用空的disease_information
    # 实际使用时应该传入正确的融合信息
    result = optimizer.optimize_diagnosis(
        user_input=symptoms,
        disease_information=[],  # 需要从调用方传入正确的信息
        initial_diagnosis=doctor_diagnosis,
        disease_list_file=disease_list_file
    )
    
    return result 