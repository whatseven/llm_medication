from src.rerank.reranker import rerank_diseases

# 测试数据 - 使用你提供的Milvus搜索结果
test_milvus_results = [
    {
        'oid': '5bb578c3831b973a137e43b9', 
        'name': '肾胚胎瘤', 
        'desc': '肾胚胎瘤又称肾母细胞瘤或Wilm\'s瘤,是幼儿时的腹内常见肿瘤，在幼儿的各种恶性肿瘤中，本病约占1/4，最多见于3岁以下的儿童，3～5岁发病率显著降低，5岁以后则少见，成人罕见，男女发病率无明显差异，多数为一侧发病，双侧同时发病者约10%左右。', 
        'symptom': '["恶心", "腹痛", "腹胀"]', 
        'similarity_score': 0.8493446111679077
    }, 
    {
        'oid': '5bb578e8831b973a137e54c6', 
        'name': '产气杆菌肠炎', 
        'desc': '产气荚膜杆菌(Clostridiumperfringen)又名魏氏梭菌(ClostridiumWelchii),它广泛存在于自然环境中，并见于几乎所有温血动物的消化道内，属于人和动物肠道内正常菌群的成员。本菌能引起人类气性坏疽及多种动物的肠毒血症和坏死性肠炎，是近年来我国家畜"猝死症"的主要病原。由产气荚膜杆菌引起的肠炎，临床上可分为食物中毒型肠炎与坏死型肠炎两种。', 
        'symptom': '["腹痛", "腹泻"]', 
        'similarity_score': 0.8307209014892578
    }, 
    {
        'oid': '5bb578db831b973a137e4edb', 
        'name': '复发性腹股沟疝', 
        'desc': '近年来大多数学者认为，腹股沟疝术后复发率在4%～10%左右。最易复发的时间在术后6～12个月以内，腹股沟直疝术后复发是斜疝的4倍，复发疝修补术后再次复发率更高。根据复发疝的发生过程，临床可将其分为遗留疝、新发疝和真性复发疝。', 
        'symptom': '["恶心", "胀痛"]', 
        'similarity_score': 0.7988762855529785
    }
]

def test_reranker():
    """测试reranker功能"""
    query_symptom = "腹痛"
    
    print("原始Milvus结果:")
    for i, result in enumerate(test_milvus_results):
        print(f"{i+1}. {result['name']} (相似度: {result['similarity_score']:.4f})")
    
    print(f"\n使用query '{query_symptom}' 进行rerank...")
    
    # 调用reranker
    reranked_results = rerank_diseases(query_symptom, test_milvus_results)
    print(reranked_results)
    
    print("\nRerank后结果:")
    for i, result in enumerate(reranked_results):
        print(f"{i+1}. {result['name']} (相关度: {result.get('relevance_score', 'N/A')})")

    
    return reranked_results

if __name__ == "__main__":
    test_reranker()
