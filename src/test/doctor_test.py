import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.model.doctor import diagnose
from src.utils.filter_disease import filter_diseases_by_name

def test_doctor_diagnose():
    """测试医生诊断功能"""
    
    # 测试数据
    user_input = "腹痛、恶心、腹胀"
    
    # 模拟向量库搜索结果
    vector_results = [
        {
            'oid': '5bb578fd831b973a137e5f5a', 
            'name': '肾胚胎瘤', 
            'desc': '肾胚胎瘤是一种儿童常见的肾脏恶性肿瘤。', 
            'symptom': '["恶心", "腹痛", "腹胀"]', 
            'similarity_score': 0.8567
        },
        {
            'oid': '5bb578ec831b973a137e56b1', 
            'name': '产气杆菌肠炎', 
            'desc': '由产气杆菌引起的肠道感染性疾病。', 
            'symptom': '["腹痛", "腹泻"]', 
            'similarity_score': 0.7234
        },
        {
            'oid': '5bb578f7831b973a137e5bc9', 
            'name': '复发性腹股沟疝', 
            'desc': '腹股沟疝的复发性疾病。', 
            'symptom': '["恶心", "胀痛"]', 
            'similarity_score': 0.6891
        }
    ]
    
    # 模拟analyzer输出的疾病列表
    analyzer_diseases = ['产气杆菌肠炎']
    
    # 测试过滤功能
    print("=== 测试疾病过滤功能 ===")
    filtered_results = filter_diseases_by_name(vector_results, analyzer_diseases)
    print(f"原始疾病数量: {len(vector_results)}")
    print(f"过滤后疾病数量: {len(filtered_results)}")
    print(f"过滤后的疾病: {[r['name'] for r in filtered_results]}")
    
    # 模拟图数据库结果
    graph_data = {
        "疾病名称": "产气杆菌肠炎",
        "疾病病因": "食物中毒型肠炎：本型产气荚膜杆菌是梭状芽胞杆菌属的种，革兰阳性，可形成芽胞。根据所产生的可溶性抗原可分为A、B、D、E等5型。食物中毒型肠炎大都由A型引起。本菌能产生各种外毒素，其中×毒素是一种卵磷脂酶，可水解卵磷脂，并有溶血作用。引起食物中毒型肠炎的肠毒素仅在形成芽胞时产生，本菌在自然界分布甚广，存在于正常人与动物粪便中，也可从土壤、垃圾、苍蝇、水、牛奶与食物中检出。",
        "检查项目": "粪细菌培养 胃肠道CT检查 胰蛋白酶", 
        "治疗科室": "传染科",
        "并发症": "代谢性酸中毒"
    }
    
    # 测试诊断功能
    print("\n=== 测试医生诊断功能 ===")
    print(f"用户输入: {user_input}")
    print(f"过滤后疾病: {[r['name'] for r in filtered_results]}")
    print(f"图数据库信息: {graph_data['疾病名称']}")
    
    print("\n=== 开始诊断 ===")
    diagnosis_result = diagnose(user_input, filtered_results, graph_data)
    print("诊断结果:")
    print(diagnosis_result)

if __name__ == "__main__":
    test_doctor_diagnose()
