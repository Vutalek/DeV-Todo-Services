import os
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import chromadb

from rag.embedding import PplxEmbedding
from .db import DBFacade
from .auth import AuthHandler

load_dotenv()

APP_DIR = Path(__file__).parent.resolve()
BASE_DIR = APP_DIR.parent

TRELLO_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BOARD_ID = os.getenv("TRELLO_BOARD_ID")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")

MODEL = "deepseek/deepseek-v4-flash"
RERANK_MODEL = "cohere/rerank-4-fast"
RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_TOP_N = 5
SIMILAR_TASKS_CONTEXT_TOP_N = 5

prompt_path = os.path.join(BASE_DIR, 'prompt', 'prompt.txt')

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY or "missing-router-api-key",
    timeout=30.0,
    max_retries=2,
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY or "missing-router-api-key",
    timeout=30.0,
    max_retries=2,
)

CHROMA_PATH = BASE_DIR / 'rag' / 'chroma_db'
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

client_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

ef = PplxEmbedding(model='perplexity/pplx-embed-v1-0.6b', client=client)
collection = client_chroma.get_or_create_collection(
    name='TODO', embedding_function=ef)

db = DBFacade()
auth = AuthHandler()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
