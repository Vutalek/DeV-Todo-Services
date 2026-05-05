import requests
from dotenv import load_dotenv
from pathlib import Path
from .embedding import PplxEmbedding
import chromadb
from openai import OpenAI
import os
from .handler_data import task_to_document, csv_to_tasks, load_tasks_to_chroma
from .bm25 import BM25TaskSearch, tasks_to_records, rrf_fusion

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / 'chroma_db'

# Initialize ChromaDB client and collection
client_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('ROUTER_API_KEY')
)

ef = PplxEmbedding(model='perplexity/pplx-embed-v1-0.6b', client=client)
collection = client_chroma.get_or_create_collection(
    name='TODO', embedding_function=ef)

          for r in reranked['results']
          ]


for item in result:
    print(
        f"{item['text']}\ntime_spent: {item['metadata']['business_days']} days")
