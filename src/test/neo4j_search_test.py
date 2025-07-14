import py2neo

# 连接到 Neo4j 数据库
# 确保 Neo4j 服务正在运行，并且端口和认证信息与你的设置一致
try:
    client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
    print("成功连接到 Neo4j 数据库！")
except Exception as e:
    print(f"连接 Neo4j 失败: {e}")
    print("请确保 Neo4j Docker 容器正在运行，并且用户名和密码正确。")
    print("你可以在 /home/ubuntu/ZJQ/NEO4J 目录下运行 'docker-compose up -d' 来启动 Neo4j 服务。")
    exit()

def search_disease_info(disease_name):
    """搜索特定疾病的详细信息"""
    print(f"\n--- 搜索疾病 '{disease_name}' 的详细信息 ---")
    query = f"""
    MATCH (n:疾病{{名称:'{disease_name}'}})
    RETURN n.名称 AS 名称, n.疾病简介 AS 简介, n.疾病病因 AS 病因,
           n.预防措施 AS 预防, n.治疗周期 AS 周期, n.治愈概率 AS 治愈率,
           n.疾病易感人群 AS 易感人群
    """
    result = client.run(query).data()
    if result:
        for record in result:
            for key, value in record.items():
                print(f"{key}: {value}")
    else:
        print(f"未找到疾病 '{disease_name}' 的信息。")

def search_related_drugs(disease_name):
    """搜索与疾病相关的药品"""
    print(f"\n--- 搜索疾病 '{disease_name}' 相关的药品 ---")
    query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病使用药品]->(drug:药品)
    RETURN drug.名称 AS 药品名称
    """
    result = client.run(query).data()
    if result:
        drugs = [record['药品名称'] for record in result]
        print(f"与 '{disease_name}' 相关的药品: {', '.join(drugs)}")
    else:
        print(f"未找到与 '{disease_name}' 相关的药品。")

def search_disease_symptoms(disease_name):
    """搜索疾病的症状"""
    print(f"\n--- 搜索疾病 '{disease_name}' 的症状 ---")
    query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病的症状]->(symptom:疾病症状)
    RETURN symptom.名称 AS 症状名称
    """
    result = client.run(query).data()
    if result:
        symptoms = [record['症状名称'] for record in result]
        print(f"'{disease_name}' 的症状: {', '.join(symptoms)}")
    else:
        print(f"未找到 '{disease_name}' 的症状。")

def search_treatment_methods(disease_name):
    """搜索疾病的治疗方法"""
    print(f"\n--- 搜索疾病 '{disease_name}' 的治疗方法 ---")
    query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:治疗的方法]->(method:治疗方法)
    RETURN method.名称 AS 治疗方法
    """
    result = client.run(query).data()
    if result:
        methods = [record['治疗方法'] for record in result]
        print(f"'{disease_name}' 的治疗方法: {', '.join(methods)}")
    else:
        print(f"未找到 '{disease_name}' 的治疗方法。")

def search_food_recommendations(disease_name):
    """搜索疾病的饮食建议 (宜吃/忌吃)"""
    print(f"\n--- 搜索疾病 '{disease_name}' 的饮食建议 ---")
    do_eat_query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病宜吃食物]->(food:食物)
    RETURN food.名称 AS 宜吃食物
    """
    no_eat_query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病忌吃食物]->(food:食物)
    RETURN food.名称 AS 忌吃食物
    """
    do_eat_result = client.run(do_eat_query).data()
    no_eat_result = client.run(no_eat_query).data()

    if do_eat_result:
        do_eat_foods = [record['宜吃食物'] for record in do_eat_result]
        print(f"'{disease_name}' 宜吃食物: {', '.join(do_eat_foods)}")
    else:
        print(f"未找到 '{disease_name}' 宜吃食物的建议。")

    if no_eat_result:
        no_eat_foods = [record['忌吃食物'] for record in no_eat_result]
        print(f"'{disease_name}' 忌吃食物: {', '.join(no_eat_foods)}")
    else:
        print(f"未找到 '{disease_name}' 忌吃食物的建议。")

def search_accompanying_diseases(disease_name):
    """搜索疾病的并发疾病"""
    print(f"\n--- 搜索疾病 '{disease_name}' 的并发疾病 ---")
    query = f"""
    MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病并发疾病]->(acompany:疾病)
    RETURN acompany.名称 AS 并发疾病
    """
    result = client.run(query).data()
    if result:
        acompanying = [record['并发疾病'] for record in result]
        print(f"'{disease_name}' 的并发疾病: {', '.join(acompanying)}")
    else:
        print(f"未找到 '{disease_name}' 的并发疾病。")

# --- 运行测试 ---
if __name__ == "__main__":
    # 你可以替换成你的知识图谱中存在的疾病名称进行测试
    test_disease = "肺放线菌病" # 假设你的知识图谱中包含“感冒”这个疾病
    # test_disease = "糖尿病"
    # test_disease = "高血压"

    search_disease_info(test_disease)
    search_related_drugs(test_disease)
    search_disease_symptoms(test_disease)
    search_treatment_methods(test_disease)
    search_food_recommendations(test_disease)
    search_accompanying_diseases(test_disease)

    print("\n--- 搜索所有疾病名称 ---")
    all_diseases_query = "MATCH (n:疾病) RETURN n.名称 AS 疾病名称 LIMIT 10"
    all_diseases_result = client.run(all_diseases_query).data()
    if all_diseases_result:
        print("部分疾病名称:")
        for record in all_diseases_result:
            print(f"- {record['疾病名称']}")
    else:
        print("未找到任何疾病。")

    print("\n--- 搜索所有药品名称 ---")
    all_drugs_query = "MATCH (n:药品) RETURN n.名称 AS 药品名称 LIMIT 10"
    all_drugs_result = client.run(all_drugs_query).data()
    if all_drugs_result:
        print("部分药品名称:")
        for record in all_drugs_result:
            print(f"- {record['药品名称']}")
    else:
        print("未找到任何药品。")
