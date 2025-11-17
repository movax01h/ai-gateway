from datetime import datetime
from http.client import responses
from typing import Annotated, Any, AsyncIterator, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, WrongUnitPrimitives
from langchain_core.messages import BaseMessage
from poetry.core.constraints.version.exceptions import ParseConstraintError
from pydantic import BaseModel, RootModel, model_validator
from starlette.responses import StreamingResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import get_prompt_registry
from ai_gateway.prompts import BasePromptRegistry, Prompt


class PromptInputs(RootModel):
    root: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def add_current_date(cls, values: dict[str, Any]) -> dict[str, Any]:
        values.setdefault("current_date", datetime.now().strftime("%B %d, %Y"))
        return values


class PromptRequest(BaseModel):
    inputs: PromptInputs
    prompt_version: Optional[str] = None
    stream: Optional[bool] = False


router = APIRouter()


def _process_chunk(chunk: BaseMessage):
    return chunk.content


@router.post(
    "/{prompt_id:path}",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
@feature_category(GitLabFeatureCategory.AI_ABSTRACTION_LAYER)
async def invoke(
    request: Request,  # pylint: disable=unused-argument
    prompt_request: PromptRequest,
    prompt_id: str,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    return await _invoke(
        prompt_request=prompt_request,
        prompt_id=prompt_id,
        current_user=current_user,
        prompt_registry=prompt_registry,
        process_chunk=_process_chunk,
        internal_event_category=__name__,
    )


async def _invoke(
    prompt_request: PromptRequest,
    prompt_id: str,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    process_chunk: Callable,
    internal_event_category: str,
):
    try:
        prompt = prompt_registry.get_on_behalf(
            current_user,
            prompt_id,
            prompt_request.prompt_version,
            internal_event_category=internal_event_category,
        )
    except ParseConstraintError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid version constraint",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_id}' not found",
        )
    except WrongUnitPrimitives:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access '{prompt_id}'",
        )

    # We don't use `isinstance` because we don't want to match subclasses
    if not type(prompt) is Prompt:  # pylint: disable=unidiomatic-typecheck
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prompt '{prompt_id}' is not supported",
        )

    try:
        if prompt_request.stream:
            response: AsyncIterator[BaseMessage] = prompt.astream(
                prompt_request.inputs.root
            )

            async def _handle_stream():
                async for chunk in response:
                    processed_chunk = process_chunk(chunk)
                    if isinstance(processed_chunk, BaseModel):
                        processed_chunk = (
                            processed_chunk.model_dump_json(exclude_none=True) + "\n"
                        )

                    yield processed_chunk

            return StreamingResponse(_handle_stream(), media_type="text/event-stream")

        response_chunk: BaseMessage = await prompt.ainvoke(prompt_request.inputs.root)
        return process_chunk(response_chunk)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        if hasattr(e, "status_code"):
            status_code = e.status_code
            err_status = f"{status_code}: {responses[status_code]}"
            raise HTTPException(
                status_code=status.HTTP_421_MISDIRECTED_REQUEST,
                # Misdirected Request https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/421
                # To signal to the client the model has produced and error
                # and not the AI Gateway
                detail=err_status,
            )
        raise e
