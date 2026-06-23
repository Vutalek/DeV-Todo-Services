import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag.handler_data import RetrievalTask
from common.deadline import calculate_deadline
from app import config
from app.config import auth, oauth2_scheme
from app.models import Message, TaskRequest, SearchRequest, create_dynamic_task_model
from app.services import rag, trello

router = APIRouter()


@router.post("/app/v1/send")
async def sendtask(message: Message, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    if not config.ROUTER_API_KEY:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": (
                    "OpenRouter task generation failed: "
                    "ROUTER_API_KEY is not set"
                ),
            },
        )

    similar_tasks = await asyncio.to_thread(
        rag.get_similar_tasks_for_message,
        message.text,
    )
    message_text = rag.build_message_text_with_similar_tasks(
        message.text,
        similar_tasks,
    )

    col_map, lab_map = await asyncio.to_thread(trello.get_trello_data)

    labels_str = ", ".join(lab_map.keys())
    columns_str = ", ".join(col_map.keys())

    Task = create_dynamic_task_model(
        columns=list(col_map.keys()),
        labels=list(lab_map.keys())
    )
    try:
        response = await config.async_client.chat.completions.parse(
            model=config.MODEL,
            messages=[
                {
                    "role": "system", "content": config.prompt.format(
                        labels_list=labels_str,
                        columns_list=columns_str
                    )
                },
                {
                    "role": "user", "content": message_text
                },
            ],
            temperature=0.1,
            top_p=0.8,
            frequency_penalty=0.3,
            max_tokens=1500,
            response_format=Task,
        )

        parsed_task = response.choices[0].message.parsed
        if parsed_task is None:
            raise ValueError("OpenRouter returned no parsed task")
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": f"OpenRouter task generation failed: {exc}",
            },
        )

    result = parsed_task.model_dump()
    result["deadline"] = calculate_deadline(parsed_task.time)

    return {"status": "success", "result": result}


@router.post("/app/v1/add_task")
def add_or_update_task(task_req: TaskRequest, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    try:
        task = RetrievalTask(
            name=task_req.name,
            desc=task_req.desc,
            prio=task_req.prio,
            label=task_req.label,
            created_at=task_req.created_at,
            finished_at=task_req.finished_at
        )

        task_id = rag.add_task_to_index(task)

        return {
            "status": "success",
            "message": f"Task '{task.name}' added successfully",
            "task_id": task_id
        }

    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": f"Failed to add task to search index: {str(e)}",
            },
        )


@router.post("/app/v1/search")
def search_tasks(search_req: SearchRequest, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    try:
        return rag.find_relevant_tasks(search_req)

    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }
