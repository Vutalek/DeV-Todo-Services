# DeV-Todo-Services

## Эндпоинты приложения

### 1. Проверка статуса сервера
```
GET /app/v1/heartbeat
```
Возвращает статус сервера

### 2. Обработка текста и создание задачи (Трелло интеграция)
```
POST /app/v1/send
Content-Type: application/json

{
  "text": "Описание задачи на естественном языке"
}
```
Проверка с тестовым JSON:
```
curl.exe -X POST -H "Content-Type: application/json" -d "@app/check.json" http://ip:port/app/v1/send
```

### 3. Добавление или обновление задачи в RAG-БД
```
POST /app/v1/tasks
Content-Type: application/json

{
  "name": "Название задачи",
  "desc": "Полное описание задачи",
  "prio": "High",
  "label": "Bug",
  "created_at": "2024-05-05T10:30:00.000000+00:00",
  "finished_at": null
}
```

**Параметры:**
- `name` (string, обязательно) - название задачи
- `desc` (string) - описание задачи
- `prio` (string, обязательно) - приоритет (High, Medium, Low и т.д.)
- `label` (string, обязательно) - метка/тип задачи (Bug, Feature, Task и т.д.)
- `created_at` (string, ISO format) - дата создания
- `finished_at` (string, ISO format) - дата завершения

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/app/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Вынести отправку webhook в отдельный клиент",
    "desc": "Сетевой вызов нужно отделить от бизнес-логики",
    "prio": "High",
    "label": "Bug",
    "created_at": "2024-05-05T10:30:00.000000+00:00"
  }'
```

**Ответ:**
```json
{
  "status": "success",
  "message": "Task 'Вынести отправку webhook в отдельный клиент' added successfully",
  "task_id": "task_0_Вынести отправку webhook"
}
```

### 4. Поиск задач в RAG-БД (гибридный поиск)
```
POST /app/v1/search
Content-Type: application/json

{
  "query": "webhook интеграция",
  "n_results": 10,
  "min_days": 0,
  "max_days": 365
}
```

**Параметры:**
- `query` (string, обязательно) - поисковый запрос
- `n_results` (integer) - количество результатов (1-50, по умолчанию 10)
- `min_days` (integer) - минимальное количество дней (по умолчанию 0)
- `max_days` (integer) - максимальное количество дней (по умолчанию 365)

**Пример запроса:**
```bash
curl -X POST "http://localhost:8000/app/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "отправка webhook",
    "n_results": 5,
    "min_days": 0,
    "max_days": 60
  }'
```

**Ответ:**
```json
{
  "status": "success",
  "query": "отправка webhook",
  "results_count": 2,
  "results": [
    {
      "task_id": "task_0_Вынести отправку webhook",
      "name": "Вынести отправку webhook в отдельный клиент",
      "description": "Сетевой вызов нужно отделить от бизнес-логики",
      "priority": "High",
      "label": "Bug",
      "created_at": "2024-05-05T10:30:00.000000+00:00",
      "finished_at": null,
      "hybrid_score": 0.85,
      "bm25_score": 3.45
    }
  ]
}
```

## RAG-система

Приложение использует гибридный поиск (BM25 + Vector Search):
- **BM25** - лексический поиск на основе ключевых слов
- **Vector Search** - семантический поиск через ChromaDB с эмбеддингами Perplexity
- **RRF Fusion** - объединение результатов обоих методов

## Проверка MCP сервиса
```
curl.exe -X POST -H "Content-Type: application/json" -d "@mcp/check.json" http://ip:port/mcp/v1/sendtask
```

## Требования
- Python 3.11+
- FastAPI
- ChromaDB
- OpenAI API ключ (OpenRouter)
- Perplexity embedding API