from dotenv import load_dotenv
from pathlib import Path
from embedding import PplxEmbedding
import chromadb
from openai import OpenAI
from handler_data import RetrievalTask
import os
from handler_data import task_to_document, csv_to_tasks, load_tasks_to_chroma
from bm25 import BM25TaskSearch, tasks_to_records, rrf_fusion
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / 'chroma_db'

client_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('ROUTER_API_KEY')
)

ef = PplxEmbedding(model='perplexity/pplx-embed-v1-0.6b', client=client)
collection = client_chroma.get_or_create_collection(
    name='TODO', embedding_function=ef)

tasks = csv_to_tasks(os.path.join(CHROMA_PATH, 'apache_issues.csv'))

ids, documents, metadatas = tasks_to_records(tasks)

# load_tasks_to_chroma(collection=collection, tasks=tasks)

query_text = task_to_document(RetrievalTask.model_validate({'name': 'Вынести отправку webhook в отдельный клиент', 'desc': 'Сетевой вызов внешнего webhook сейчас выполняется прямо внутри бизнес-логики завершения счета, из-за чего усложняется сопровождение и снижается изоляция интеграционного кода. Нужно выделить отправку webhook в отдельный клиент, чтобы отделить интеграцию от доменной логики и упростить дальнейшее развитие обработки внешних уведомлений.', 'prio': 'Highest', 'label': 'Bug'}
                                                           ))

bm25 = BM25TaskSearch(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)

bm25_results = bm25.search(query_text, n_results=10, where_days=(0, 60))

result = collection.query(
    query_texts=[query_text], n_results=10, include=['metadatas', 'distances'], where={
        '$and': [
            {'business_days': {'$gte': 0}},
            {'business_days': {'$lte': 60}},
        ]
    })

hybrid_results = rrf_fusion(
    vector_results=result,
    bm25_results=bm25_results,
    top_n=3,
)

for item in hybrid_results:
    print(item['hybrid_score'], item['metadata'])
