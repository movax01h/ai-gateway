from fastapi import APIRouter

from ai_gateway.api.v1.embeddings import code_embeddings

__all__ = [
    "router",
]


router = APIRouter()

router.include_router(code_embeddings.router)
