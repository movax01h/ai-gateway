import json
import logging
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action

# Setup logger
logger = logging.getLogger(__name__)


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


class GitlabHttpClient:
    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata

    async def aget(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        parse_json: bool = True,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        return await self._call(
            path, "GET", parse_json, params=params, object_hook=object_hook
        )

    async def apost(self, path: str, body: str, parse_json: bool = True) -> Any:
        return await self._call(path, "POST", parse_json, data=body)

    async def aput(self, path: str, body: str, parse_json: bool = True) -> Any:
        return await self._call(path, "PUT", parse_json, data=body)

    async def apatch(self, path: str, body: str, parse_json: bool = True) -> Any:
        return await self._call(path, "PATCH", parse_json, data=body)

    async def _call(
        self,
        path: str,
        method: str,
        parse_json: bool = True,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        object_hook: Union[Callable, None] = None,
    ):
        if params:
            query_string = urlencode(params)
            path = f"{path}?{query_string}"

        response = await _execute_action(
            self.metadata,
            contract_pb2.Action(
                runHTTPRequest=contract_pb2.RunHTTPRequest(
                    path=path, method=method, body=data
                )
            ),
        )

        if not parse_json:
            return response

        try:
            return json.loads(response, object_hook=object_hook)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {method} {path}: {str(e)}. ")
            logger.error(
                f"Raw response type: {type(response)}, content: {repr(response)}"
            )

            return {}
