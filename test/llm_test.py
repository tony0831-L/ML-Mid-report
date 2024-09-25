# 導入套件
from transformers import AutoTokenizer, AutoModelForCausalLM


# Part 2 範例
# @model :hugginface 模型
# @max_length: 最大長度
def llm_test(model: str, max_length: int, prompt: str) -> str:
    # 加載標記器
    tokenizer = AutoTokenizer.from_pretrained(model)
    # 加載預訓練模型
    model = AutoModelForCausalLM.from_pretrained(model)

    # 使用標記器處理文本
    user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # 生成回應
    response_hf_encoded = model.generate(
        user_input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )

    # 處理回應, 只保留生成部分, 並將生成的回應轉成可讀字串
    response_hf: str = tokenizer.decode(

        response_hf_encoded[:, user_input_ids.shape[-1]:][0],
        skip_special_tokens=True

    )

    # 返回回應
    return response_hf