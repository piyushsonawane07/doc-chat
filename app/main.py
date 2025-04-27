import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os
from langchain_community.document_loaders import PyPDFLoader
from app.tasks import process_pdf_page
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from celery.result import AsyncResult
from app.tasks import celery_app
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PERSIST_DIR = "./chroma_langchain_db"


class QueryRequest(BaseModel):
    filename: str
    question: str


class PreviewRequest(BaseModel):
    filename: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    if os.path.exists(file_path):
        print("File already exists")
    else:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        if not os.path.exists(file_path):
            return {"error": "Failed to save uploaded file"}

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    total_pages = len(docs)

    task_ids = [] 

    for page_num in range(total_pages):
        print("Page Content: ", docs[page_num].page_content)
        page_content = docs[page_num].page_content
        metadata = docs[page_num].metadata
        metadata['source'] = file.filename

        task = process_pdf_page.delay(page_content, metadata, page_num)
        task_ids.append(task.id) 

    return {
        "message": f"Started processing {total_pages} pages.",
        "task_ids": task_ids  
    }


@app.get("/status/{task_id}")
async def check_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status
    }


@app.get("/files")
async def list_uploaded_files():
    files = os.listdir(UPLOAD_DIR)
    return {"uploaded_files": files}


@app.delete("/delete/{filename}")
async def delete_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    os.remove(file_path)
    return {"message": f"{filename} deleted successfully."}


@app.get("/collections")
async def list_chroma_collections():
    vector_store = Chroma(
        collection_name="doc-chat-collection",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=PERSIST_DIR,
    )
    return {"collections": vector_store.get()["ids"]}


@app.post("/query")
async def query_document(request: QueryRequest):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="doc-chat-collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"source": {"$eq": request.filename}}
        }
    )

    prompt_template = """You are a helpful assistant that answers questions based on the provided context. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    llm = ChatOpenAI(model="gpt-4o",temperature=0)
    
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(request.question)

    return {"answer": answer}


@app.get("/preview/{filename}")
async def preview_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"'
    }
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
        headers=headers
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
