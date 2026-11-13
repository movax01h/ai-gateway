"""The REST route table that the AI Gateway serves."""

from fastapi import APIRouter, FastAPI

__all__ = ["mount_routes"]


def mount_routes(app: FastAPI) -> None:
    """Mount the v1-v4 API routers and the monitoring router on ``app``.

    Args:
        app: The FastAPI application to add the route table to.
    """
    # ai_gateway.api.server imports this module, so these imports cannot be top-level.
    # pylint: disable=import-outside-toplevel
    from ai_gateway.api.monitoring import router as http_monitoring_router
    from ai_gateway.api.v1 import api_router as http_api_router_v1
    from ai_gateway.api.v2 import api_router as http_api_router_v2
    from ai_gateway.api.v3 import api_router as http_api_router_v3
    from ai_gateway.api.v4 import api_router as http_api_router_v4

    # pylint: enable=import-outside-toplevel

    sub_router = APIRouter()
    sub_router.include_router(http_api_router_v1, prefix="/v1")
    sub_router.include_router(http_api_router_v2, prefix="/v2")
    sub_router.include_router(http_api_router_v3, prefix="/v3")
    sub_router.include_router(http_api_router_v4, prefix="/v4")
    sub_router.include_router(http_monitoring_router)

    app.include_router(sub_router)
