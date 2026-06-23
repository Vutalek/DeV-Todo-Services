from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.rag import load_tasks_from_chroma
from app.routes import health, auth, projects, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_tasks_from_chroma()
    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "*",
    "http://localhost",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(tasks.router)
