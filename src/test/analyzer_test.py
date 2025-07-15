import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.model.analyzer import analyze_diagnosis

def test_diagnosis_analyzer():
    """测试医疗诊断分析器"""
    
    # 测试数据
    user_input = "我最近总是腹痛，还有点恶心，有时候会腹胀"
    
    test_disease_results = [{'oid': '5bb578e8831b973a137e54c6', 'name': '产气杆菌肠炎', 'desc': '产气荚膜杆菌(Clostridiumperfringen)又名魏氏梭菌(ClostridiumWelchii),它广泛存在于自然环境中，并见于几乎所有温血动物的消化道内，属于人和动物肠道内正常菌群的成员。本菌能引起人类气性坏疽及多种动物的肠毒血症和坏死性肠炎，是近年来我国家畜"猝死症"的主要病原。由产气荚膜杆菌引起的肠炎，临床上可分为食物中毒型肠炎与坏死型肠炎两种。', 'symptom': '["腹痛", "腹泻"]', 'similarity_score': 0.8307209014892578, 'relevance_score': 0.6918097138404846}, {'oid': '5bb578c3831b973a137e43b9', 'name': '肾胚胎瘤', 'desc': "肾胚胎瘤又称肾母细胞瘤或Wilm's瘤,是幼儿时的腹内常见肿瘤，在幼儿的各种恶性肿瘤中，本病约占1/4，最多见于3岁以下的儿童，3～5岁发病率显著降低，5岁以后则少见，成人罕见，男女发病率无明显差异，多数为一侧发病，双侧同时发病者约10%左右。", 'symptom': '["恶心", "腹痛", "腹胀"]', 'similarity_score': 0.8493446111679077, 'relevance_score': 0.543832540512085}, {'oid': '5bb578db831b973a137e4edb', 'name': '复发性腹股沟疝', 'desc': '近年来大多数学者认为，腹股沟疝术后复发率在4%～10%左右。最易复发的时间在术后6～12个月以内，腹股沟直疝术后复发是斜疝的4倍，复发疝修补术后再次复发率更高。根据复发疝的发生过程，临床可将其分为遗留疝、新发疝和真性复发疝。', 'symptom': '["恶心", "胀痛"]', 'similarity_score': 0.7988762855529785, 'relevance_score': 0.028007520362734795}]
    
    print("=== 医疗诊断分析器测试 ===")
    print(f"患者症状: {user_input}")
    print("\n候选疾病:")
    for disease in test_disease_results:
        print(f"- {disease['name']}: {disease['symptom']}")
    
    # 调用分析器
    result = analyze_diagnosis(user_input, test_disease_results)
    print(result)
    
    print(f"\n分析结果: {result}")

if __name__ == "__main__":
    test_diagnosis_analyzer()
