from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import auth, db, oauth2_scheme

router = APIRouter()


@router.post("/app/v1/create_project")
async def create_project(name: str, description: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.create_project(login, name, description)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}


@router.get("/app/v1/projects")
async def get_projects(token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.get_user_projects(login)
    result = [
        {
            "id": str(entry[0]),
            "name": entry[1],
            "description": entry[2]
        }
    for entry in result]
    return {"status": "success", "result": result}


@router.get("/app/v1/project_members")
async def get_project_members(project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.get_project_members(login, project)
    result = [
        {
            "id": str(entry[0]),
            "login": entry[1],
            "role": entry[2]
        }
    for entry in result]
    return {"status": "success", "result": result}


@router.get("/app/v1/add_member")
async def add_member(member_login: str, project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.add_member_to_project(login, member_login, project)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}


@router.post("/app/v1/leave_project")
async def leave_project(project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.leave_project(login, project)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}


@router.delete("/app/v1/delete_project")
async def delete_project(project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.delete_project(login, project)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}
