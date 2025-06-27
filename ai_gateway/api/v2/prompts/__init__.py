from fastapi import APIRouter

from ai_gateway.api.v2.prompts import invoke

__all__ = [
    "router",
]


router = APIRouter()
router.include_router(invoke.router)
