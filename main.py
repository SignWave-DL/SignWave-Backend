from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from db.database import engine, Base
from Api.ws import router as ws_router
from Api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: auto-create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown code here if needed

app = FastAPI(title="Audio → Whisper → Gloss → Postgres", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(api_router)
