import os
from dotenv import load_dotenv
from interface.gpt_info import gpt_info
from test.gpt_test import gpt_test
from test.langchain_rag_test import langchain_rag_test
from test.llm_test import llm_test


def main():
    load_dotenv()

    # Part 1 - Using LLMs in Python Using API
    info: gpt_info = gpt_info("gpt-4o-mini", 0, 5, 0)
    gpt_resp: str = gpt_test(os.getenv('apikey'), info, "你叫鄭培宇")
    print("gpt_reps: " + gpt_resp + "\n")

    # Part 2 - Using Open Source LLMs Locally
    llm_resp = llm_test(os.getenv('llmModel'), 1000, "你叫鄭培宇")
    print("llm_resp: " + llm_resp + "\n")

    # Part 3 - Setting up LangChain configurations and pipeline
    ragResp: str = langchain_rag_test()
    print("ragResp: " + ragResp + "\n")


if __name__ == '__main__':
    main()
