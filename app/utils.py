# utils.py
import openai
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def store_document(documents, uuids):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="doc-chat-collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    
    vector_store.add_documents(documents=documents, ids=uuids)
