# 導入套件
import os
from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# Part 3 範例
def langchain_rag_test():
    # 指定用甚麼字符串分割文本
    split_text_by = '"Title: Mocked up record'
    # 文本塊大小
    chunk_size = 2000

    # 文本不重疊
    chunk_overlap = 0

    # 載入文本
    text_loader = TextLoader(os.getenv('ragCsvPath'), encoding="utf-8")
    # 保存加載后的文本
    documents = text_loader.load()

    # 配置文本分割器
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=split_text_by)

    # 分割.csv檔中內容的字符串
    splitted_docs = text_splitter.split_documents(documents)

    # 載入嵌入式模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 創建向量資料庫
    vector_db = FAISS.from_documents(splitted_docs, embeddings)

    # 搜索字符
    query1 = "Which patients is an office worker"

    # 搜索結果
    docs = vector_db.similarity_search(query1)

    # 返回搜索結果(生成可讀字串)
    return textwrap.fill(

        # 搜索結果中的第一個文檔的內容
        str(docs[0].page_content),

        # 最大寬度100 ( textwrap.fill的配置項 )
        width=100,

        # 不替換空白 ( textwrap.fill的配置項 )
        replace_whitespace=False

    )