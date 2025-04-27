# Smart Document Q&A System

A scalable, efficient document question-answering system built with FastAPI, Celery, OpenAI, and Chroma.  
Upload PDFs, ask natural language questions, and get smart, accurate answers — optimized for large document handling with distributed async processing.

---

## Features

- Upload and manage PDF documents
- Asynchronous, distributed page-wise processing with Celery
- Chunk and store documents using LangChain and Chroma vector store
- Natural Language Understanding with OpenAI's GPT-3.5
- Intelligent semantic search over document content
- Simple REST APIs for file management, querying, and collection browsing
- Clean architecture with proper logging and modular tasks

---

## Tech Stack

- **FastAPI** — Modern, fast (high-performance) web framework
- **Celery** — Distributed task queue for asynchronous processing
- **RabbitMQ** — Message broker for Celery
- **Chroma** — Local vector database for semantic document storage
- **LangChain** — Framework for building with LLMs
- **OpenAI API** — For embeddings and answering queries
- **Uvicorn** — ASGI server for FastAPI
- **aiofiles** — Asynchronous file handling

---

## Project Structure

```
app/
  ├── main.py        # FastAPI app with endpoints
  ├── tasks.py       # Celery tasks for async page processing
  ├── utils.py       # Utility functions (e.g., store documents)
  ├── logs/          # Celery log files
uploaded_documents/  # Uploaded PDFs
chroma_langchain_db/ # Vector database storage
.env                 # Environment variables (e.g., OpenAI API key)
README.md            # Project documentation
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start RabbitMQ**

   Ensure RabbitMQ is running locally or update `BROKER_URL` in `tasks.py`.

5. **Run Celery Worker**
   ```bash
   celery -A app.tasks.celery_app worker --loglevel=info
   ```

6. **Run the FastAPI App**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Environment Variables**

   Create a `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

---

## API Endpoints

| Method | Endpoint                   | Description                          |
|:------:|:---------------------------:|:------------------------------------:|
| POST   | `/upload`                   | Upload a new PDF file               |
| GET    | `/files`                    | List all uploaded PDFs              |
| DELETE | `/delete/{filename}`         | Delete a specific uploaded PDF      |
| GET    | `/collections`              | List available document collections |
| POST   | `/query`                    | Query a document for answers        |
| GET    | `/status/{task_id}`          | Check Celery task status            |

---

## Example Usage

- **Upload a PDF**
  ```bash
  POST /upload
  Form-Data: file=<your-pdf-file>
  ```

- **Ask a Question**
  ```bash
  POST /query
  Body:
  {
    "filename": "example.pdf",
    "question": "What is the summary of the document?"
  }
  ```

- **Check Task Status**
  ```bash
  GET /status/{task_id}
  ```

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Celery](https://docs.celeryq.dev/en/stable/)
- [LangChain](https://www.langchain.dev/)
- [Chroma](https://docs.trychroma.com/)
- [OpenAI](https://openai.com/)
- [RabbitMQ](https://www.rabbitmq.com/)

---

## Like this project?

If you found this useful, give it a ⭐️ and share it with others!

---
