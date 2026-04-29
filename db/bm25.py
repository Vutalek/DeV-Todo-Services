from rank_bm25 import BM25Okapi
import re
from parse_data import RetrievalTask, task_to_document, task_to_metadata


class BM25TaskSearch:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas

        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 10, where_days: tuple[int, int] | None = None):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        candidates = range(len(scores))

        if where_days is not None:
            min_days, max_days = where_days
            candidates = [
                i for i in candidates
                if min_days <= self.metadatas[i].get('business_days', 0) <= max_days
            ]

        ranked_indexes = sorted(
            candidates,
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        return [
            {
                'id': self.ids[i],
                'document': self.documents[i],
                'metadata': self.metadatas[i],
                'bm25_score': float(scores[i]),
            }
            for i in ranked_indexes
        ]


def rrf_fusion(vector_results, bm25_results, k: int = 60, top_n: int = 10):
    scores = {}
    payloads = {}

    for rank, doc_id in enumerate(vector_results['ids'][0], start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        payloads[doc_id] = {
            'id': doc_id,
            'metadata': vector_results['metadatas'][0][rank - 1],
            'vector_distance': vector_results['distances'][0][rank - 1],
            'source': 'vector',
        }

    for rank, item in enumerate(bm25_results, start=1):
        doc_id = item['id']
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        if doc_id not in payloads:
            payloads[doc_id] = {
                'id': doc_id,
                'metadata': item['metadata'],
                'bm25_score': item['bm25_score'],
                'source': 'bm25',
            }
        else:
            payloads[doc_id]['bm25_score'] = item['bm25_score']
            payloads[doc_id]['source'] = 'hybrid'

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]

    return [
        {
            **payloads[doc_id],
            'hybrid_score': scores[doc_id],
        }
        for doc_id in ranked_ids
    ]


def tasks_to_records(tasks: list[RetrievalTask]):
    ids = []
    documents = []
    metadatas = []

    for i, task in enumerate(tasks):
        ids.append(f'task_{i}')
        documents.append(task_to_document(task))
        metadatas.append(task_to_metadata(task))

    return ids, documents, metadatas


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())
