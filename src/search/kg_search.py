import py2neo
from typing import List, Dict, Any

def search_diseases_by_symptoms(symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    根据症状搜索相关疾病信息
    
    Args:
        symptoms: 症状列表，如 ["胸痛", "呼吸困难"]
        
    Returns:
        List[Dict]: 最多5个疾病信息，按症状匹配数量降序排列
                   每个字典包含: name, desc, symptom, acompany, cause, cure_department
    """
    try:
        # 连接Neo4j数据库
        client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
        
        # 构建Cypher查询
        # 通过症状节点查找相关疾病
        query = """
        MATCH (d:疾病)-[:疾病的症状]->(s:疾病症状)
        WHERE s.名称 IN $symptoms
        WITH d, COUNT(s) AS match_count, COLLECT(s.名称) AS matched_symptoms
        RETURN d.名称 AS name,
               d.疾病简介 AS desc,
               d.疾病病因 AS cause,
               d.预防措施 AS prevent,
               d.治疗周期 AS cure_lasttime,
               d.治愈概率 AS cured_prob,
               d.疾病易感人群 AS easy_get,
               matched_symptoms AS symptom,
               match_count
        ORDER BY match_count DESC
        LIMIT 3
        """
        
        # 执行查询
        result = client.run(query, symptoms=symptoms).data()
        
        # 格式化返回结果
        diseases = []
        for record in result:
            disease_name = record.get('name', '')
            
            # 查询治疗科室
            dept_query = """
            MATCH (d:疾病{名称: $disease_name})-[:疾病所属科目]->(dept:科目)
            RETURN COLLECT(dept.名称) AS departments
            """
            dept_result = client.run(dept_query, disease_name=disease_name).data()
            departments = dept_result[0]['departments'] if dept_result else []
            
            # 查询并发症
            comp_query = """
            MATCH (d:疾病{名称: $disease_name})-[:疾病并发疾病]->(comp:疾病)
            RETURN COLLECT(comp.名称) AS complications
            """
            comp_result = client.run(comp_query, disease_name=disease_name).data()
            complications = comp_result[0]['complications'] if comp_result else []
            
            disease = {
                'name': disease_name,
                'desc': record.get('desc', ''),
                'symptom': record.get('symptom', []),
                'cause': record.get('cause', ''),
                'cure_department': departments,
                'acompany': complications
            }
            diseases.append(disease)
        
        return diseases
        
    except Exception as e:
        print(f"知识图谱症状搜索错误: {e}")
        return []
