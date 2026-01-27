import functools
import typing

from dependency_injector import providers
from fastapi import Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel

from ai_gateway.api.middleware import X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER
from ai_gateway.async_dependency_resolver import get_config

__all__ = ["ChatInvokable", "authorize_with_unit_primitive"]


class ChatInvokable(BaseModel):
    name: str
    unit_primitive: GitLabUnitPrimitive


def authorize_with_unit_primitive(
    request_param: str, *, chat_invokables: list[ChatInvokable]
):
    def decorator(func: typing.Callable) -> typing.Callable:
        chat_invokable_by_name = {ci.name: ci for ci in chat_invokables}

        @functools.wraps(func)
        async def wrapper(
            request: Request,
            config: typing.Annotated[providers.Configuration, Depends(get_config)],
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            request_param_val = request.path_params[request_param]

            chat_invokable = chat_invokable_by_name.get(request_param_val, None)
            if not chat_invokable:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
                )

            current_user = request.user
            unit_primitive = chat_invokable.unit_primitive
            if not current_user.can(unit_primitive):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unauthorized to access {unit_primitive}",
                )

            if (
                request.headers.get(X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER)
                == "duo_core"
                and config.process_level_feature_flags.duo_classic_chat_duo_core_cutoff()
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Duo Core no longer authorized to access Duo Classic Chat",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
