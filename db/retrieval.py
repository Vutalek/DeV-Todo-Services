import requests
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

bm25_results = bm25.search(query_text, n_results=30, where_days=(0, 60))

vectors_result = collection.query(
    query_texts=[query_text], n_results=30, include=['distances'], where={
        '$and': [
            {'business_days': {'$gte': 0}},
            {'business_days': {'$lte': 60}},
        ]
    })

hybrid_results = rrf_fusion(
    vector_results=vectors_result,
    bm25_results=bm25_results,
    top_n=30,
)

documents_id = [item['id'] for item in hybrid_results]
chroma_data = collection.get(ids=documents_id)

docs = chroma_data['documents']
metadatas = chroma_data['metadatas']

response = requests.post(
    'https://openrouter.ai/api/v1/rerank',
    headers={
        'Authorization': f"Bearer {os.getenv('ROUTER_API_KEY')}",
        'Content-Type': 'application/json'
    },
    json={
        'model': 'cohere/rerank-4-fast',
        'query': query_text,
        'documents': docs,
        'top_n': 5
    }
)

reranked = response.json()
result = [{'text': r['document']['text'], 'metadata': metadatas[r['index']]}
          for r in reranked['results']
          ]


for item in result:
    print(
        f"{item['text']}\ntime_spent: {item['metadata']['business_days']} days")
