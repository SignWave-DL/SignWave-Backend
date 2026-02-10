from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Api.ws import router as ws_router

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
        app.include_router(ws_router)