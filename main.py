from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database import engine, Base
from Api.ws import router as ws_router
from Api.routes import router as api_router

app = FastAPI(title="Audio → Whisper → Gloss → Postgres")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # auto-create tables (simple demo)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(ws_router)
app.include_router(api_router)
