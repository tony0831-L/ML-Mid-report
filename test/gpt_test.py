from typing import List

import openai
import re
import time

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from interface.gpt_info import gpt_info


def gpt_test(api_key: str, info: gpt_info, system_prompt: str) -> ChatCompletion:
    mem_list = []
    client = openai.OpenAI(api_key=api_key)

    mem_list.append(
        {
            "role": "system",
            "content": system_prompt
        }
    )

    response: ChatCompletion = client.chat.completions.create(
        model=info.openai_model,
        messages=mem_list,
        temperature=info.temperature
    )

    return response.choices[0].message.content
