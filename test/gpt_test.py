# 導入套件
from typing import Optional

import openai
from openai.types.chat import ChatCompletion
from interface.gpt_info import gpt_info


# Part 1 範例
# @api_key :API金鑰
# @info: gpt設定
# @system_prompt: 預設系統prompt
def gpt_test(api_key: str, info: gpt_info, system_prompt: str) -> Optional[str]:
    # 記憶陣列
    mem_list = []

    # openai客戶端
    client = openai.OpenAI(api_key=api_key)

    # 新增系統預設prompt
    mem_list.append(
        {
            "role": "system",
            "content": system_prompt
        }
    )

    # 取得回應
    response: ChatCompletion = client.chat.completions.create(
        model=info.openai_model,
        messages=mem_list,
        temperature=info.temperature
    )

    # 返回回應
    return response.choices[0].message.content
