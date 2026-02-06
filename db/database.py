import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Use SQLite for local development (or PostgreSQL if remote)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./app.db"  # Local SQLite database
)

engine = create_async_engine(DATABASE_URL, echo=False, future=True, connect_args={"timeout": 10} if "sqlite" in DATABASE_URL else {})
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class Base(DeclarativeBase):
    pass

async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
