# tasks.py
import uuid
from celery import Celery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils import store_document
from langchain.schema import Document
import logging
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Configure Celery logger
logger = logging.getLogger('celery')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'celery.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

celery_app = Celery(
    "tasks",
    broker="amqp://localhost",
    backend="rpc://"
)

@celery_app.task
def process_pdf_page(page_content: str, metadata: dict, page_number: int):
    logger.info(f"Starting to process page {page_number}")
    try:
        if 'source' not in metadata:
            metadata['source'] = "unknown.pdf"

        doc = Document(page_content=page_content, metadata=metadata)
        all_splits = text_splitter.split_documents([doc])
        valid_splits = [split for split in all_splits if split.page_content.strip()]
        uuids = [str(uuid.uuid4()) for _ in range(len(valid_splits))]

        if valid_splits:
            store_document(valid_splits, uuids)
            logger.info(f"Stored page {page_number} successfully")
        else:
            logger.warning(f"No valid content found on page {page_number}, skipped storing.")

    except Exception as e:
        logger.error(f"Error processing page {page_number}: {str(e)}", exc_info=True)
        raise
