from fastapi import APIRouter
from app.api.routes import ingest, vectorstore, chat

api_router = APIRouter()

api_router.include_router(ingest.router, prefix="/ingest/v1", tags=["ingest"])
api_router.include_router(vectorstore.router, prefix="/vectorstore/v1", tags=["vectorstore"])
api_router.include_router(chat.router, prefix="/chat/v1", tags=["chat"])
