from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


@register_embedding_function
class PplxEmbedding(EmbeddingFunction):

    def __init__(self, model, client):
        self.model = model
        self.client = client
        self.batch_size = 256

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            if response is None:
                raise RuntimeError('API returned None')

            if not getattr(response, 'data', None):
                raise RuntimeError(
                    f'No embedding data received'
                    f'{start}:{start + len(batch)}'
                )

            batch_embeddings = [item.embedding for item in response.data]

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f'Embedding count mismatch: got {len(batch_embeddings)}, '
                    f'expected {len(batch)} for batch {start}:{start + len(batch)}'
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @staticmethod
    def name() -> str:
        return 'PplxEmbd'

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> 'EmbeddingFunction':
        return PplxEmbedding(config['model'], config['client'])
