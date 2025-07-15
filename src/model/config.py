# 大模型API配置
MODELS = {
    "deepseek": {
        "api_key": "sk-ad96b1f3603c4e6e8769a52c38bcbcd7",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "qwen": {
        "api_key": "sk-your-qwen-api-key", 
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-turbo"
    },
    "doubao": {
        "api_key": "sk-your-doubao-api-key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3", 
        "model_name": "doubao-lite-4k"
    }
}

# 默认使用的模型
DEFAULT_MODEL = "deepseek"
