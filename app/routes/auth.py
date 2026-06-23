from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.config import auth, db
from app.models import Token

router = APIRouter()


@router.post("/register")
async def register(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    exists = await db.get_user_id(form_data.username)
    if exists != "":
        raise HTTPException(status_code=409, detail="User already exists")
    is_created = await db.create_user(form_data.username, form_data.password)
    if not is_created:
        raise HTTPException(status_code=500, detail="Failed to create user")

    token = auth.create_access_token(data={"sub": form_data.username})
    return Token(access_token=token, token_type="bearer")


@router.post("/token")
async def token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    is_authenticated = await auth.authenticate_user(
        form_data.username,
        form_data.password,
    )
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth.create_access_token(data={"sub": form_data.username})
    return Token(access_token=token, token_type="bearer")
