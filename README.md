# DeV-Todo-Services

Сервис превращает TODO/описание задачи в структурированную Trello-задачу и использует RAG-поиск по историческим задачам для подбора похожего контекста.

## Основные сервисы

- `app/app.py` - основной FastAPI-сервис.
- `db/chroma_db` - persistent ChromaDB storage.

## Переменные окружения

```env
ROUTER_API_KEY=...
TRELLO_API_KEY=...
TRELLO_TOKEN=...
TRELLO_BOARD_ID=...
TRELLO_LIST_ID=...
```

- `ROUTER_API_KEY` используется для OpenRouter chat completions, embeddings и reranker.
- `TRELLO_API_KEY`, `TRELLO_TOKEN`, `TRELLO_BOARD_ID` нужны `/app/v1/send`, чтобы получить списки и labels Trello.

## App API

### Heartbeat

```http
GET /app/v1/heartbeat
```

Ответ:

```json
{
  "status": "alive"
}
```

Важно: `heartbeat` проверяет только то, что приложение запущено. Он не проверяет готовность Chroma, BM25, Trello или OpenRouter.

### Генерация задачи

```http
POST /app/v1/send
Content-Type: application/json
```

Тело:

```json
{
  "text": "TODO: вынести отправку webhook в отдельный клиент"
}
```

`message.text` может содержать результат `/app/v1/search` со списком похожих задач. Модель использует похожие задачи как контекст для выбора `label`, `prio`, `time` и формулировки новой задачи.

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

- `time` - количество рабочих часов до дедлайна.
- `deadline` вычисляется сервером по рабочему календарю: будни, 10:00-18:00, timezone `Europe/Moscow`.
- При ошибке OpenRouter ручка возвращает HTTP `502` с JSON `{"status": "error", "message": "..."}`.

Проверка:

```bash
curl -X POST "http://localhost:8000/app/v1/send" \
  -H "Content-Type: application/json" \
  -d @app/check.json
```

### Добавление задачи в RAG-БД

```http
POST /app/v1/tasks
Content-Type: application/json
```

Тело:

```json
{
  "name": "Вынести отправку webhook в отдельный клиент",
  "desc": "Сетевой вызов нужно отделить от бизнес-логики",
  "prio": "High",
  "label": "Bug",
  "created_at": "2024-05-05T10:30:00.000000+00:00",
  "finished_at": "2024-05-05T12:30:00.000000+00:00"
}
```

Поля:

- `name` - название задачи.
- `desc` - описание задачи.
- `prio` - исторический приоритет задачи.
- `label` - исторический тип/label задачи.
- `created_at` - дата создания в ISO format.
- `finished_at` - дата завершения в ISO format.

Успешный ответ:

```json
{
  "status": "success",
  "message": "Task 'Вынести отправку webhook в отдельный клиент' added successfully",
  "task_id": "task_..."
}
```

При записи считается metadata:

- `business_days` - рабочая длительность в днях.
- `lead_time_hours` - рабочая длительность в часах.

Ночи и выходные не считаются рабочим временем.

### Поиск похожих задач

```http
POST /app/v1/search
Content-Type: application/json
```

Тело:

```json
{
  "query": "отправка webhook",
  "n_results": 10,
  "min_days": 0,
  "max_days": 365
}
```

Pipeline:

1. BM25 lexical search.
2. Chroma vector search через OpenRouter embeddings.
3. RRF fusion.
4. OpenRouter reranker `cohere/rerank-4-fast`.
5. Возврат top 5 задач.

Успешный ответ:

```json
{
  "status": "success",
  "query": "отправка webhook",
  "results_count": 5,
  "results": [
    {
      "name": "Вынести отправку webhook в отдельный клиент",
      "desc": "Сетевой вызов нужно отделить от бизнес-логики",
      "priority": "High",
      "label": "Bug",
      "reranker_score": 0.87,
      "business_days": 0.5,
      "time_hours": 4
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

## RAG Chroma volume

При старте приложение открывает Chroma collection `TODO`.

В Docker deploy используется named volume:

```bash
app-chroma-db:/app/db/chroma_db
```

Volume хранит ChromaDB между деплоями.

## Docker deploy

Workflow `.github/workflows/main.yml`:

1. Собирает image `app`.
2. Создаёт persistent volume `app-chroma-db`.
3. Запускает `app-new` на временном порту `8001`.
4. Проверяет `GET /app/v1/heartbeat`.
5. Только после успешного healthcheck останавливает старый `app`.
6. Запускает новый `app` на порту `8000`.
7. Повторно проверяет healthcheck.

Это защищает от ситуации, когда новый image не стартует, а старый контейнер уже остановлен.

## Быстрые проверки

```bash
python3 -m compileall app db common
```

```bash
uv run pytest -q
```

```bash
uv run python -c "from fastapi.testclient import TestClient; import app.app as mod; client = TestClient(mod.app); print(client.get('/app/v1/heartbeat').json(), len(mod.tasks_store), mod.collection.count())"
```

## Требования

- Python 3.11+
- FastAPI
- ChromaDB
- OpenRouter API key
- Trello API key/token
