from contextlib import asynccontextmanager

from fastapi import FastAPI

from application.router import router
from core.db.mongo import client, db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.command("ping")
    yield
    await client.close()


app = FastAPI(lifespan=lifespan)

app.include_router(router)
