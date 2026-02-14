from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Api.ws import router as ws_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    app.include_router(ws_router)
    yield
    # Shutdown logic (if needed)

app = FastAPI(
    title="Audio → Whisper → Gloss → Postgres",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)