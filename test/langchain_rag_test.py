import os
from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def langchain_rag_test():
    split_text_by = '"Title: Mocked up record'
    chunk_size = 2000
    chunk_overlap = 0

    text_loader = TextLoader(os.getenv('ragCsvPath'), encoding="utf-8")
    documents = text_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=split_text_by)
    splitted_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_db = FAISS.from_documents(splitted_docs, embeddings)

    query1 = "Which patients is an office worker"
    docs = vector_db.similarity_search(query1)
    return textwrap.fill(str(docs[0].page_content), width=100, replace_whitespace=False)
