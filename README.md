# DeV-Todo-Services

FastAPI-сервис для превращения свободного TODO-текста в структурированную Trello-задачу. Перед генерацией сервис ищет похожие исторические задачи в RAG-базе, добавляет их в контекст модели и на этой основе помогает выбрать `label`, `prio`, `time`, `roadmap` и колонку Trello.

## Что внутри

- `app/app.py` - основной FastAPI API.
- `prompt/prompt.txt` - системный prompt для генерации Trello-задачи.
- `db/handler_data.py` - модель исторической задачи и расчет рабочих часов.
- `db/bm25.py` - lexical search и RRF fusion.
- `db/embedding.py` - OpenRouter embedding function для ChromaDB.
- `db/chroma_db` - persistent ChromaDB storage.
- `common/deadline.py` - расчет дедлайна по рабочему календарю.
- `benchmark/` - отдельный runner для сравнения LLM-моделей через OpenRouter.
- `tests/` - unit/API тесты.

## Pipeline

1. Исторические задачи добавляются в RAG-БД через `/app/v1/add_task`.
2. При старте приложение поднимает Chroma collection `TODO` и восстанавливает BM25 index из Chroma.
3. `/app/v1/search` ищет похожие задачи:
   - BM25 lexical search;
   - Chroma vector search;
   - RRF fusion;
   - OpenRouter reranker `cohere/rerank-4-fast`;
   - фильтр `reranker_score > 0.5`, если reranker доступен.
4. `/app/v1/send` автоматически вызывает этот же поиск по `Message.text`, добавляет найденные задачи в user-context и отправляет обогащенный текст в OpenRouter chat completion.
5. Ответ модели валидируется динамической Pydantic-моделью, построенной из актуальных Trello колонок и labels.
6. `deadline` считается сервером из `time` по рабочему календарю: будни, 10:00-18:00, timezone `Europe/Moscow`.

## Переменные окружения

Создай `.env` по примеру `.env.example`:

```env
ROUTER_API_KEY=...
TRELLO_API_KEY=...
TRELLO_TOKEN=...
```

- `ROUTER_API_KEY` нужен для OpenRouter chat completions, embeddings и reranker.
- `TRELLO_API_KEY`, `TRELLO_TOKEN`, `TRELLO_BOARD_ID` нужны `/app/v1/send`, чтобы получить актуальные списки и labels с доски.
- `TRELLO_LIST_ID` может встречаться в CI secrets как legacy-переменная, но текущий код использует список колонок доски, а не фиксированный list id.

## Локальный запуск

```bash
uv sync
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:

```bash
curl http://localhost:8000/app/v1/heartbeat
```

Ответ:

```json
{
  "status": "alive"
}
```

`heartbeat` проверяет только то, что приложение запущено. Он не проверяет готовность Chroma, BM25, Trello или OpenRouter.

## API

### GET `/app/v1/heartbeat`

Healthcheck приложения.

```json
{
  "status": "alive"
}
```

### POST `/app/v1/add_task`

Добавляет историческую задачу в ChromaDB и обновляет in-memory BM25 index.

```json
{
  "name": "Вынести отправку webhook в отдельный клиент",
  "desc": "Сетевой вызов нужно отделить от бизнес-логики",
  "prio": "High",
  "label": "Bug",
  "created_at": "2024-05-05T10:30:00+03:00",
  "finished_at": "2024-05-05T12:30:00+03:00"
}
```

Успешный ответ:

```json
{
  "status": "success",
  "message": "Task 'Вынести отправку webhook в отдельный клиент' added successfully",
  "task_id": "task_..."
}
```

При записи рассчитываются:

- `business_days` - рабочая длительность в днях;
- `lead_time_hours` - рабочая длительность в часах.

Ночи и выходные не считаются рабочим временем.

### POST `/app/v1/search`

Ищет похожие задачи в RAG-БД.

```json
{
  "query": "отправка webhook",
  "n_results": 10,
  "min_days": 0,
  "max_days": 365
}
```

Успешный ответ:

```json
{
  "status": "success",
  "query": "отправка webhook",
  "results_count": 1,
  "results": [
    {
      "name": "Вынести отправку webhook в отдельный клиент",
      "desc": "Сетевой вызов нужно отделить от бизнес-логики",
      "priority": "High",
      "label": "Bug",
      "reranker_score": 0.87,
      "business_days": 0.25,
      "time_hours": 2
    }
  ]
}
```

Если reranker недоступен, `/search` не падает: возвращает top 5 из hybrid search, добавляет `warning`, а `reranker_score` будет `null`.

Если база пустая:

```json
{
  "status": "warning",
  "message": "No tasks in database",
  "results": []
}
```

### POST `/app/v1/send`

Генерирует структурированную задачу из свободного текста. Похожий контекст вручную передавать не нужно: endpoint сам делает поиск по `text` и добавляет в prompt похожие задачи со всеми полями:

- `name`
- `desc`
- `priority`
- `label`
- `reranker_score`
- `business_days`
- `time_hours`

Запрос:

```json
{
  "text": "TODO: вынести отправку webhook в отдельный клиент"
}
```

Успешный ответ:

```json
{
  "status": "success",
  "result": {
    "name": "Вынести отправку webhook в отдельный клиент",
    "desc": "Сейчас отправка webhook смешана с бизнес-логикой. Нужно вынести сетевой вызов в отдельный клиент.",
    "label": ["Bug"],
    "prio": 3,
    "time": 4,
    "roadmap": "Шаг 1 (1 часов): найти текущий вызов webhook\nШаг 2 (2 часов): вынести вызов в клиент\nШаг 3 (1 часов): проверить обработку ошибок",
    "column": "Backlog",
    "deadline": "2026-05-20T15:00:00+00:00"
  }
}
```

При ошибке Trello/OpenRouter endpoint возвращает HTTP `502` с JSON:

```json
{
  "status": "error",
  "message": "..."
}
```

## Примеры curl

```bash
curl -X POST "http://localhost:8000/app/v1/add_task" \
  -H "Content-Type: application/json" \
  -d @app/check_task.json
```

```bash
curl -X POST "http://localhost:8000/app/v1/search" \
  -H "Content-Type: application/json" \
  -d @app/check_search.json
```

```bash
curl -X POST "http://localhost:8000/app/v1/send" \
  -H "Content-Type: application/json" \
  -d @app/check.json
```

## Тесты и проверки

```bash
uv run pytest
```

```bash
python3 -m compileall app db common benchmark
```

Быстрая проверка FastAPI без внешнего HTTP-сервера:

```bash
uv run python -c "from fastapi.testclient import TestClient; import app.app as mod; client = TestClient(mod.app); print(client.get('/app/v1/heartbeat').json(), len(mod.tasks_store), mod.collection.count())"
```

## Docker

Локальная сборка:

```bash
docker build -t app .
```

Запуск:

```bash
docker volume create app-chroma-db
docker run -d \
  --name app \
  --env-file .env \
  -v app-chroma-db:/app/db/chroma_db \
  -p 8000:8000 \
  app
```

Dockerfile запускает:

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Один worker важен, потому что BM25 index живет в памяти процесса.

## Deploy

Workflow `.github/workflows/main.yml` запускается на push в `main` и вручную через `workflow_dispatch`.

Что делает deploy:

1. Копирует код на сервер через SSH.
2. Создает `.env` из GitHub Secrets.
3. Собирает Docker image `app`.
4. Создает persistent volume `app-chroma-db`.
5. Запускает временный контейнер `app-new` на порту `8001`.
6. Проверяет `GET /app/v1/heartbeat`.
7. После успешного healthcheck заменяет production container `app` на порту `8000`.
8. Повторно проверяет healthcheck и чистит старые образы.

Volume `app-chroma-db:/app/db/chroma_db` хранит ChromaDB между деплоями.

## Benchmark

`benchmark/` позволяет прогонять примеры из JSONL через несколько candidate-моделей и оценивать ответы judge-моделями.

```bash
uv run python benchmark/main.py \
  --input benchmark/data/examples.jsonl \
  --output-dir benchmark/res
```

Результаты пишутся в:

- `benchmark/res/results.jsonl`
- `benchmark/res/summary.json`

Модели и prompts по умолчанию лежат в `benchmark/config.py` и `benchmark/prompts/`.

## Требования

- Python `>=3.11,<3.14`
- FastAPI
- ChromaDB
- OpenRouter API key
- Trello API key/token/board id
