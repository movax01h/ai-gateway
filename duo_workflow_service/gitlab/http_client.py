import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Setup logger
logger = logging.getLogger(__name__)


class GitLabHttpResponse:
    def __init__(self, status_code, body, headers=[]):
        self.status_code = status_code
        self.body = body
        self.headers = headers


def checkpoint_decoder(json_object: dict):
    if not ("type" in json_object and "content" in json_object):
        return json_object

    message_type = json_object.pop("type")
    if message_type == "SystemMessage":
        return SystemMessage(**json_object)
    elif message_type == "HumanMessage":
        return HumanMessage(**json_object)
    elif message_type == "AIMessage":
        return AIMessage(**json_object)
    elif message_type == "ToolMessage":
        return ToolMessage(**json_object)
    else:
        json_object["type"] = message_type
        return json_object


class GitlabHttpClient(ABC):
    """Abstract base class defining the interface for GitLab HTTP clients."""

    async def aget(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        parse_json: bool = True,
        use_http_response: bool = False,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        return await self._call(
            path,
            "GET",
            parse_json=parse_json,
            use_http_response=use_http_response,
            params=params,
            object_hook=object_hook,
        )

    async def apost(
        self,
        path: str,
        body: str,
        parse_json: bool = True,
        use_http_response: bool = False,
    ) -> Any:
        return await self._call(
            path,
            "POST",
            parse_json=parse_json,
            use_http_response=use_http_response,
            data=body,
        )

    async def aput(
        self,
        path: str,
        body: str,
        parse_json: bool = True,
        use_http_response: bool = False,
    ) -> Any:
        return await self._call(
            path,
            "PUT",
            parse_json=parse_json,
            use_http_response=use_http_response,
            data=body,
        )

    async def apatch(
        self,
        path: str,
        body: str,
        parse_json: bool = True,
        use_http_response: bool = False,
    ) -> Any:
        return await self._call(
            path,
            "PATCH",
            parse_json=parse_json,
            use_http_response=use_http_response,
            data=body,
        )

    def _parse_response(
        self,
        response: Any,
        parse_json: bool = True,
        use_http_response: bool = False,
        object_hook: Union[Callable, None] = None,
    ) -> Union[Dict[str, Any], list, str, None]:
        """Parse the response from the API call.

        Args:
            response: The raw response (string or other data)
            parse_json: Whether to parse the response as JSON
            object_hook: Optional JSON decoder hook for custom object deserialization

        Returns:
            Parsed response data (dict/list) if parsing succeeds,
            or raw response (str) if parsing fails or is not requested,
            or None if response is None
        """
        if not parse_json:
            return response

        try:
            if isinstance(response, str):
                if object_hook:
                    return json.loads(response, object_hook=object_hook)
                return json.loads(response)
            elif isinstance(response, (dict)):
                return response  # Already parsed JSON (dict)

            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}. ")
            logger.error(
                f"Raw response type: {type(response)}, content: {repr(response)}"
            )
            return {}

    @abstractmethod
    async def _call(
        self,
        path: str,
        method: str,
        parse_json: bool = True,
        use_http_response: bool = False,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        pass
