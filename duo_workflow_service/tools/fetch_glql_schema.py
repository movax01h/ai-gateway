"""Read the GLQL schema from the instance.

Supersedes `get_glql_schema`, which serves a bundled deprecated copy for
instances older than the GitLab 19.3 endpoint, that do not have the /schema API.
"""

import asyncio
import json
from typing import Any, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, PrivateAttr

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.version_compatibility import (
    GLQL_SCHEMA_ENDPOINT_VERSION,
    supports_glql_schema_endpoint,
)
from lib.context import gitlab_version

log = structlog.stdlib.get_logger("fetch_glql_schema")

SCHEMA_PATH = "/api/v4/glql/schema"
# Kept below DuoBaseTool's 200 KiB output truncation floor, so an oversized
# schema fails here with a clear error instead of being silently cut mid-JSON.
MAX_SCHEMA_BYTES = 192 * 1024


class FetchGlqlSchemaInput(BaseModel):
    """No arguments: the endpoint takes none and returns the schema whole."""


class FetchGlqlSchema(DuoBaseTool):
    """Serve the GLQL schema the instance publishes."""

    name: str = "fetch_glql_schema"
    description: str = """Get the GLQL schema.

    Returns every GLQL data source with its query modes, each mode's own filter,
    display, sort, dimension, metric and parameterized field lists, and the
    vocabularies needed to read them: operators, value_kinds, reference_types,
    functions and display_types.

    MUST be called before building any GLQL query, so that only fields the
    instance actually supports are used. Call it ONCE per question: it returns
    everything, and takes no arguments.

    Available on GitLab 19.3 and later.
    """
    args_schema: Type[BaseModel] = FetchGlqlSchemaInput

    # Cached for the lifetime of the tool instance, which is one workflow run:
    # the tool registry is rebuilt per ExecuteWorkflow run, so this dedups
    # calls within a turn. The schema only changes on instance upgrade anyway.
    _body: Optional[str] = PrivateAttr(default=None)
    # Held while fetching, so batched parallel calls make one request instead
    # of one each. Built on first use rather than eagerly: two locks never
    # compare equal, and pydantic equality covers private attributes, so an
    # eager one would make two freshly built tools unequal.
    _fetching: Optional[asyncio.Lock] = PrivateAttr(default=None)

    async def _execute(self, *_args: Any, **_kwargs: Any) -> str:
        # Nothing is awaited between the check and the assignment, so callers
        # cannot end up with a lock each.
        if self._fetching is None:
            self._fetching = asyncio.Lock()

        async with self._fetching:
            if self._body is None:
                self._body = await self._fetch()

        return self._body

    def _check_version(self) -> None:
        """Say why the schema is unavailable instead of returning a bare 404."""
        if supports_glql_schema_endpoint():
            return

        raise ToolException(
            "The GLQL schema endpoint is only available in GitLab "
            f"{GLQL_SCHEMA_ENDPOINT_VERSION} and later. "
            f"Current GitLab version: {gitlab_version.get() or 'unknown'}"
        )

    async def _fetch(self) -> str:
        self._check_version()

        response = await self.gitlab_client.aget(path=SCHEMA_PATH, parse_json=False)
        body = self._process_http_response(SCHEMA_PATH, response, log)

        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if not isinstance(body, str):
            body = json.dumps(body)

        if len(body.encode("utf-8")) > MAX_SCHEMA_BYTES:
            raise ToolException(
                f"GLQL schema response too large: over {MAX_SCHEMA_BYTES} bytes "
                "(the base tool would truncate anything bigger mid-document)."
            )

        try:
            document = json.loads(body)
        except json.JSONDecodeError as error:
            raise ToolException(
                f"GLQL schema response was not valid JSON: {error}"
            ) from error

        if not isinstance(document, dict) or "sources" not in document:
            raise ToolException("GLQL schema response did not contain any sources.")

        return body

    def format_display_message(
        self, _args: FetchGlqlSchemaInput, _tool_response: Any = None
    ) -> str:
        return "Looking up the GLQL schema"
