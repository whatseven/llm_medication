import py2neo

def neo4j_disease_search(name: str) -> str:
    """
    根据疾病名称搜索Neo4j中的疾病信息
    
    Args:
        name: 疾病名称
        
    Returns:
        str: 格式化的疾病信息文本，适合大模型阅读
    """
    try:
        # 连接Neo4j数据库
        client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
        
        # 查询疾病基本信息（节点属性）
        disease_query = f"""
        MATCH (n:疾病{{名称:'{name}'}})
        RETURN n.疾病病因 AS 病因, n.预防措施 AS 预防, n.治疗周期 AS 周期
        """
        disease_result = client.run(disease_query).data()
        
        if not disease_result:
            return ""
        
        disease_info = disease_result[0]
        
        # 查询常用药品
        drug_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病使用药品]->(drug:药品)
        RETURN drug.名称 AS 药品名称
        """
        drug_result = client.run(drug_query).data()
        drugs = [record['药品名称'] for record in drug_result]
        
        # 查询宜吃食物
        do_eat_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病宜吃食物]->(food:食物)
        RETURN food.名称 AS 食物名称
        """
        do_eat_result = client.run(do_eat_query).data()
        do_eat_foods = [record['食物名称'] for record in do_eat_result]
        
        # 查询忌吃食物
        not_eat_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病忌吃食物]->(food:食物)
        RETURN food.名称 AS 食物名称
        """
        not_eat_result = client.run(not_eat_query).data()
        not_eat_foods = [record['食物名称'] for record in not_eat_result]
        
        # 查询检查项目
        check_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病所需检查]->(check:检查项目)
        RETURN check.名称 AS 检查名称
        """
        check_result = client.run(check_query).data()
        checks = [record['检查名称'] for record in check_result]
        
        # 查询治疗科室
        department_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病所属科目]->(dept:科目)
        RETURN dept.名称 AS 科室名称
        """
        department_result = client.run(department_query).data()
        departments = [record['科室名称'] for record in department_result]
        
        # 查询治疗方法
        cure_way_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:治疗的方法]->(method:治疗方法)
        RETURN method.名称 AS 治疗方法
        """
        cure_way_result = client.run(cure_way_query).data()
        cure_ways = [record['治疗方法'] for record in cure_way_result]
        
        # 查询并发症
        acompany_query = f"""
        MATCH (d:疾病{{名称:'{name}'}})-[:疾病并发疾病]->(acompany:疾病)
        RETURN acompany.名称 AS 并发疾病
        """
        acompany_result = client.run(acompany_query).data()
        acompanies = [record['并发疾病'] for record in acompany_result]
        
        # 格式化输出
        result_text = f"疾病名称：{name}\n\n"
        
        # 疾病病因（前200字符）
        cause = disease_info.get('病因', '')
        if cause:
            cause_text = cause[:200] if len(cause) > 200 else cause
            result_text += f"疾病病因：{cause_text}\n\n"
        
        # 预防措施
        prevent = disease_info.get('预防', '')
        if prevent:
            result_text += f"预防措施：{prevent}\n\n"
        
        # 治疗周期
        cycle = disease_info.get('周期', '')
        if cycle:
            result_text += f"治疗周期：{cycle}\n\n"
        
        # 治疗科室
        if departments:
            result_text += f"治疗科室：{' '.join(departments)}\n\n"
        
        # 治疗方法
        if cure_ways:
            result_text += f"治疗方法：{' '.join(cure_ways)}\n\n"
        
        # 检查项目
        if checks:
            result_text += f"检查项目：{' '.join(checks)}\n\n"
        
        # 常用药品
        if drugs:
            result_text += f"常用药品：{' '.join(drugs)}\n\n"
        
        # 宜吃食物
        if do_eat_foods:
            result_text += f"宜吃食物：{' '.join(do_eat_foods)}\n\n"
        
        # 忌吃食物
        if not_eat_foods:
            result_text += f"忌吃食物：{' '.join(not_eat_foods)}\n\n"
        
        # 并发症
        if acompanies:
            result_text += f"并发症：{' '.join(acompanies)}\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        print(f"Neo4j查询错误: {e}")
        return ""
