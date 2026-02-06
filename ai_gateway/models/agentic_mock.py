"""Agentic mock model that allows responses to be specified in the request. The model will extract responses and tool
calls from tags, enabling multi-node graph execution.

Usage examples:
    Basic response:
        "Summarize https://gitlab.com/example-issue. <response>Here is the summary: The issue discusses...</response>"

    Multi-node workflow with tool calls (must be an array):
        "Analyze the issue.
        <response>
            I need to search for more info then I'll provide analysis.
            <tool_calls>[{"name": "search", "args": {"query": "issue details"}}]</tool_calls>
        </response>"

    Sequential responses for multiple agent calls:
        "Do complex task.
        <responses>
            <response>I'll start by gathering information. <tool_calls>[{"name": "get_info"}]</tool_calls></response>
            <response>Based on the info, here's my analysis...</response>
        </responses>"

    Response with latency simulation:
        "<response latency_ms='500'>This response will be delayed by 500ms</response>"

    Response with streaming simulation:
        "<response stream='true' chunk_delay_ms='50'>This response will be streamed token by token</response>"

    Response from file:
        "Analyze this code. <response file='tests/fixtures/long_response.txt' stream='true' chunk_delay_ms='50' />"

    Template variable substitution (Jinja2):
    Define variables using {% set %} statements, then use {{ variable }} placeholders in responses.

    Example goal:
        {% set projectId = 1000001 %}
        {% set namespace = "my-namespace" %}

        Test the project.
        <response file='fixtures/list_repo_tree_tool_call.txt' />

    Example file content (fixtures/list_repo_tree_tool_call.txt):
        Let me examine the current project structure to understand the codebase.
        <tool_calls>
        [
            {
                "name": "gitlab_api_get",
                "args": {
                    "endpoint": "/api/v4/projects/{{ projectId }}/repository/tree",
                    "params": { "per_page": 100 }
                }
            }
        ]
        </tool_calls>

    The {{ projectId }} will be replaced with 1000001 when the file is loaded.
"""

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, NamedTuple, Optional

import structlog
from jinja2 import Environment
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

log = structlog.stdlib.get_logger("agentic_mock")


class Response(NamedTuple):
    content: str
    tool_calls: list[ToolCall] = []
    latency_ms: int = 0
    stream: bool = False
    chunk_delay_ms: int = 0


class ResponseHandler:
    """Handles parsing and state management for scripted responses."""

    # Class-level cache for file contents to ensure each file is only loaded once
    _file_cache: dict[str, str] = {}

    def __init__(self, messages: list[BaseMessage], use_last_human_message: bool):
        self.content, ai_messages = self._get_user_input(
            messages, use_last_human_message
        )
        if self.content:
            self.content = self._strip_additional_context(self.content)
        self.template_vars = self._extract_template_variables()
        self.jinja_env = self._create_jinja_environment()
        all_responses = self._parse_all_responses() if self.content else []
        self.responses: list[Response] = [
            response
            for response in all_responses
            if not self._is_responded_already(response, ai_messages)
        ]

        self.current_index = 0

    def _get_user_input(
        self, messages: list[BaseMessage], use_last_human_message: bool
    ) -> tuple[Optional[str], Optional[list[AIMessage]]]:
        if not messages:
            return None, None

        ai_messages: list[AIMessage] = []

        if use_last_human_message:
            user_message = None

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    ai_messages = []
                elif isinstance(msg, AIMessage):
                    ai_messages.append(msg)
        else:
            user_message = next(
                (msg for msg in messages if isinstance(msg, HumanMessage)), None
            )

        if not user_message or not user_message.content:
            return None, None

        return user_message.text(), ai_messages

    def _is_responded_already(
        self, response: Response, ai_messages: Optional[list[AIMessage]]
    ):
        if not ai_messages:
            return False

        for msg in ai_messages:
            if (
                msg.content == response.content
                and msg.tool_calls == response.tool_calls
            ):
                return True

        return False

    def _parse_all_responses(self) -> list[Response]:
        """Parse all defined responses from the user input content, return as a list."""
        assert self.content is not None

        if "<response" not in self.content:
            return [Response(content="mock response (no response tag specified)")]

        log.debug("Parsing responses from user input", content_length=len(self.content))

        parsed_responses = []

        # Wrap content in a root element to ensure valid XML
        wrapped_xml = f"<root>{self._strip_jinja_syntax(self.content)}</root>"

        try:
            root = ET.fromstring(wrapped_xml)
            response_elements = root.findall(".//response")

            log.debug("Found response elements", count=len(response_elements))

            for idx, response_element in enumerate(response_elements):
                attributes = response_element.attrib

                file_path = attributes.get("file")
                if file_path:
                    response_text = self._load_response_from_file(file_path)
                    response_source = "file"
                else:
                    response_text = self._extract_text_from_element(response_element)
                    response_source = "inline"

                clean_content, tool_calls = self._extract_tools_from_response_text(
                    response_text
                )

                latency_ms = self._parse_int_attribute(attributes, "latency_ms", 0)
                stream = self._parse_bool_attribute(attributes, "stream", False)
                chunk_delay_ms = self._parse_int_attribute(
                    attributes, "chunk_delay_ms", 0
                )

                log.debug(
                    "Parsed response",
                    index=idx,
                    response_source=response_source,
                    content_length=len(clean_content),
                    tool_calls_count=len(tool_calls),
                    latency_ms=latency_ms,
                    stream=stream,
                    chunk_delay_ms=chunk_delay_ms,
                )

                parsed_responses.append(
                    Response(
                        content=clean_content,
                        tool_calls=tool_calls,
                        latency_ms=latency_ms,
                        stream=stream,
                        chunk_delay_ms=chunk_delay_ms,
                    )
                )

        except ET.ParseError as e:
            log.error("Failed to parse responses as XML", error=str(e))
            return [Response(content="mock response (invalid xml)")]

        return parsed_responses

    def _extract_template_variables(self) -> dict[str, Any]:
        """Extract template variables from {% set %} statements by rendering the template and capturing the context.

        Supports Jinja2 set statements like:
        {% set projectId = 1000001 %}
        {% set namespace = "my-namespace" %}

        Returns a dictionary mapping variable names to their values.
        """
        template_vars: dict[str, Any] = {}

        if not self.content:
            return template_vars

        try:
            env = Environment()
            template = env.from_string(self.content)

            # Render the template as a module to capture all variables set via {% set %}
            # The module's __dict__ contains all variables defined in the template
            module = template.make_module()

            # Extract variables from the module's namespace
            # Filter out internal Jinja variables (those starting with '_')
            template_vars = {
                key: value
                for key, value in module.__dict__.items()
                if not key.startswith("_")
            }

            log.debug("Extracted template variables", template_vars=template_vars)

        except Exception as e:
            log.warning("Failed to extract template variables", error=str(e))

        return template_vars

    def _create_jinja_environment(self) -> Environment:
        return Environment(
            autoescape=False,  # Don't escape content since we're dealing with JSON/code
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _parse_tool_calls_json(
        self, tool_json_text: str, tool_calls: list[ToolCall]
    ) -> None:
        """Parse and validate tool calls JSON, appending valid tool calls to the list.

        Args:
            tool_json_text: JSON string containing tool call definitions
            tool_calls: List to append parsed tool calls to

        Raises:
            ValueError: If JSON is invalid or tool calls don't match expected structure
        """
        try:
            tools_json = json.loads(tool_json_text.strip())
            if not isinstance(tools_json, list):
                raise ValueError(
                    f"Tool calls must be an array, got {type(tools_json).__name__}: {tool_json_text.strip()}"
                )

            for tool in tools_json:
                if not isinstance(tool, dict):
                    raise ValueError(
                        f"Each tool call must be an object, got {type(tool).__name__}: {tool}"
                    )
                if "name" not in tool:
                    raise ValueError(f"Tool call missing required 'name' field: {tool}")

                tool_call: ToolCall = {
                    "name": tool["name"],
                    "args": tool.get("args", {}),
                    "id": f"call_{len(tool_calls) + 1}",
                    "type": "tool_call",
                }
                tool_calls.append(tool_call)
                log.debug(
                    "Parsed tool call", tool_name=tool["name"], tool_id=tool_call["id"]
                )

        except json.JSONDecodeError as e:
            log.error(
                "Invalid JSON in tool_calls",
                tool_json=tool_json_text.strip(),
                error=str(e),
            )
            raise ValueError(
                f"Invalid JSON in tool_calls: {tool_json_text.strip()}"
            ) from e

    def _extract_tools_from_response_text(
        self, response_text: str
    ) -> tuple[str, list[ToolCall]]:
        """Extract tool calls from response text and remove the tool_calls tags.

        For file-based responses, we need to handle the content carefully since it may
        contain XML special characters (< > &) that would break XML parsing.

        Returns:
            A tuple of (clean_content, tool_calls) where clean_content has all
            <tool_calls>...</tool_calls> sections removed.
        """
        if "<tool_calls>" not in response_text:
            return response_text.strip(), []

        tool_calls: list[ToolCall] = []
        clean_parts = []
        remaining_text = response_text

        while "<tool_calls>" in remaining_text:
            start_idx = remaining_text.find("<tool_calls>")
            end_idx = remaining_text.find("</tool_calls>")

            if end_idx == -1:
                log.warning("Found <tool_calls> opening tag without closing tag")
                clean_parts.append(remaining_text[:start_idx])
                break

            # Extract the part before the tool_calls tag (this is the clean text)
            clean_parts.append(remaining_text[:start_idx])

            # Extract and parse the JSON content between the tags
            tool_json_text = remaining_text[start_idx + len("<tool_calls>") : end_idx]
            self._parse_tool_calls_json(tool_json_text, tool_calls)

            # Continue with the remaining text after the closing tag
            # (skip the entire <tool_calls>...</tool_calls> section)
            remaining_text = remaining_text[end_idx + len("</tool_calls>") :]

        # Add any remaining text after the last tool_calls tag
        clean_parts.append(remaining_text)
        clean_response = "".join(clean_parts).strip()

        log.debug(
            "Tool extraction complete",
            total_tool_calls=len(tool_calls),
            clean_content_length=len(clean_response),
        )

        return clean_response, tool_calls

    def _extract_text_from_element(self, element: ET.Element) -> str:
        """Recursively extract text content from an XML element and its children.

        Note: This extracts ALL text including <tool_calls> tags which will be
        processed later by _extract_tools_from_response_text to separate content
        from tool calls.
        """
        text_parts = []

        if element.text:
            text_parts.append(element.text)

        for child in element:
            # Special handling for tool_calls - we need to preserve the XML structure
            # so it can be extracted and removed later
            if child.tag == "tool_calls":
                text_parts.append("<tool_calls>")
                if child.text:
                    text_parts.append(child.text)
                text_parts.append("</tool_calls>")
            else:
                text_parts.append(self._extract_text_from_element(child))

            if child.tail:
                text_parts.append(child.tail)

        return "".join(text_parts)

    def _parse_int_attribute(
        self, attributes: dict[str, str], key: str, default: int
    ) -> int:
        """Parse an integer attribute value with a default."""
        value = attributes.get(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            log.warning("Invalid integer attribute value", key=key, value=value)
            return default

    def _parse_bool_attribute(
        self, attributes: dict[str, str], key: str, default: bool
    ) -> bool:
        """Parse a boolean attribute value with a default."""
        value = attributes.get(key)
        if value is None:
            return default

        return value.lower() == "true"

    def _load_response_from_file(self, file_path: str) -> str:
        """Load response content from a file.

        Files are cached to ensure each is only loaded once.
        """
        if file_path in self._file_cache:
            log.debug("Loading response from cache", file_path=file_path)
            content = self._file_cache[file_path]
        else:
            path = Path(file_path)
            if not path.exists():
                log.error("Response file not found", file_path=file_path)
                raise FileNotFoundError(f"Response file not found: {file_path}")

            content = path.read_text(encoding="utf-8")
            self._file_cache[file_path] = content
            log.debug(
                "Loaded and cached response file",
                file_path=file_path,
                content_length=len(content),
            )

        content = self._render_template(content)
        return content

    def _strip_jinja_syntax(self, content: str) -> str:
        """Remove Jinja template syntax ({% ... %}) from content.

        This is needed because template variables are extracted separately, and the {% %} syntax breaks XML parsing.
        """
        content = re.sub(r"{%.*?%}", "", content, flags=re.DOTALL)
        return content

    def _strip_additional_context(self, content: str) -> str:
        """Remove <additional_context> sections added by the system.

        The system may inject additional context with content that breaks XML parsing. Since agentic mock doesn't need
        this context, we strip it out.
        """
        # Remove everything from "User added additional context" through the closing tag
        # This includes the intro text, opening tag, content, and closing tag
        content = re.sub(
            r"User added additional context[^\n]*\n+<additional_context>.*?</additional_context>\s*\n*",
            "",
            content,
            flags=re.DOTALL,
        )
        return content

    def _render_template(self, content: str) -> str:
        """Render content as a Jinja2 template with template_vars as context.

        Supports:
        - {{ projectId }} - replaced with projectId from context
        - {{ namespace }} - replaced with namespace from context
        - Any other variables in template_vars
        """
        if not self.template_vars:
            return content

        try:
            template = self.jinja_env.from_string(content)
            result = template.render(**self.template_vars)
            log.debug("Rendered template", vars_used=list(self.template_vars.keys()))
            return result
        except Exception as e:
            log.error(
                "Failed to render template", error=str(e), content_preview=content[:100]
            )
            return content

    def get_next_response(self) -> Response:
        """Get the next response in sequence, returning empty response when exhausted."""
        if not self.content:
            log.debug("No content available, returning default response")
            return Response(content="mock response (no response tag specified)")

        if self.current_index >= len(self.responses):
            log.debug(
                "All scripted responses exhausted",
                current_index=self.current_index,
                total_responses=len(self.responses),
            )
            return Response(content="mock response (all scripted responses exhausted)")

        response = self.responses[self.current_index]
        log.debug(
            "Returning next response",
            current_index=self.current_index,
            total_responses=len(self.responses),
            content_length=len(response.content),
            tool_calls_count=len(response.tool_calls),
        )
        self.current_index += 1
        return response


class AgenticFakeModel(BaseChatModel):
    """Mock model for agentic workflows that extracts responses and tool calls from tags in the input.

    Supports:
    - Basic responses: <response>text</response>
    - Tool calls for multi-node execution: <tool_calls>[{"name": "tool", "args": {}}]</tool_calls>
    - Sequential responses: multiple <response> tags in order
    """

    # Disable LangChain caching for this model so multiple responses are treated as such. Otherwise only the
    # first response will be handled.
    cache: bool = False

    def __init__(
        self, auto_tool_approval: bool, use_last_human_message: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self._response_handler: Optional[ResponseHandler] = None
        self._auto_tool_approval = auto_tool_approval
        self._use_last_human_message = use_last_human_message

    @property
    def _is_agentic_mock_model(self) -> bool:
        return True

    @property
    def _is_auto_approved_by_agentic_mock_model(self) -> bool:
        return self._auto_tool_approval

    @property
    def _llm_type(self) -> str:
        return "agentic-fake-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "agentic-fake-model"}

    async def _generate_with_latency(self, messages: list[BaseMessage]) -> ChatResult:
        if self._response_handler is None:
            self._response_handler = ResponseHandler(
                messages, use_last_human_message=self._use_last_human_message
            )

        response = self._response_handler.get_next_response()

        if response.latency_ms > 0:
            log.debug("Applying latency", latency_ms=response.latency_ms)
            await asyncio.sleep(response.latency_ms / 1000.0)

        ai_message = AIMessage(
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else [],
        )

        log.debug(
            "Generated response",
            content_length=len(response.content),
            tool_calls_count=len(response.tool_calls),
        )

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        result = asyncio.run(self._generate_with_latency(messages))

        # Invoke callbacks for LangSmith tracing
        if run_manager:
            llm_result = LLMResult(generations=[[gen] for gen in result.generations])
            run_manager.on_llm_end(llm_result)

        return result

    def bind_tools(self, *_args: Any, **_kwargs: Any) -> Any:
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await self._generate_with_latency(messages)

        # Invoke callbacks for LangSmith tracing
        if run_manager:
            llm_result = LLMResult(generations=[[gen] for gen in result.generations])
            await run_manager.on_llm_end(llm_result)

        return result

    def _generate_text_chunks(
        self, response: Response
    ) -> Iterator[tuple[ChatGenerationChunk, str]]:
        """Generate text content chunks for streaming."""
        words = response.content.split()
        log.debug(
            "Streaming response",
            word_count=len(words),
            chunk_delay_ms=response.chunk_delay_ms,
        )

        for i, word in enumerate(words):
            # Add space before word except for the first one
            chunk_text = word if i == 0 else f" {word}"
            chunk = AIMessageChunk(content=chunk_text)
            yield ChatGenerationChunk(message=chunk), chunk_text

    def _generate_tool_call_chunks(
        self, response: Response
    ) -> Iterator[ChatGenerationChunk]:
        """Generate tool call chunks for streaming."""
        log.debug(
            "Streaming tool calls in chunks", tool_calls_count=len(response.tool_calls)
        )

        for tool_call in response.tool_calls:
            tool_json = json.dumps(
                {
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call["id"],
                    "type": tool_call["type"],
                }
            )

            chunk_size = 100

            for i in range(0, len(tool_json), chunk_size):
                chunk_text = tool_json[i : i + chunk_size]

                # For LangChain, the first chunk should include the tool call structure,
                # subsequent chunks update the args
                if i == 0:
                    name = tool_call["name"]
                    args = chunk_text if len(tool_json) <= chunk_size else ""
                else:
                    name = None
                    args = chunk_text

                tool_chunk = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": name,
                            "args": args,
                            "id": tool_call["id"],
                            "index": 0,
                        }
                    ],
                )

                yield ChatGenerationChunk(message=tool_chunk)

    def _generate_stream_chunks(
        self, response: Response
    ) -> Iterator[tuple[ChatGenerationChunk, Optional[str]]]:
        """Generate all chunks for streaming (both text and tool calls).

        Yields tuples of (chunk, token_text) where:
        - chunk: The ChatGenerationChunk to yield
        - token_text: Text token for callback manager (None for tool calls)
        """
        if response.stream:
            for chunk_generation, chunk_text in self._generate_text_chunks(response):
                yield chunk_generation, chunk_text

            if response.tool_calls:
                for chunk_generation in self._generate_tool_call_chunks(response):
                    yield chunk_generation, None
        else:
            log.debug("Yielding complete non-streaming response")
            chunk_generation = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=response.content,
                    tool_calls=response.tool_calls if response.tool_calls else [],
                )
            )
            yield chunk_generation, response.content

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response synchronously."""
        if self._response_handler is None:
            self._response_handler = ResponseHandler(
                messages, use_last_human_message=self._use_last_human_message
            )
        response = self._response_handler.get_next_response()

        log.debug(
            "Starting stream",
            stream_enabled=response.stream,
            latency_ms=response.latency_ms,
        )

        # Apply initial latency
        if response.latency_ms > 0:
            time.sleep(response.latency_ms / 1000.0)

        for chunk, token_text in self._generate_stream_chunks(response):
            if run_manager and token_text:
                run_manager.on_llm_new_token(token_text)
            yield chunk
            if response.chunk_delay_ms > 0:
                time.sleep(response.chunk_delay_ms / 1000.0)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the response asynchronously."""
        if self._response_handler is None:
            self._response_handler = ResponseHandler(
                messages, use_last_human_message=self._use_last_human_message
            )
        response = self._response_handler.get_next_response()

        log.debug(
            "Starting async stream",
            stream_enabled=response.stream,
            latency_ms=response.latency_ms,
        )

        # Apply initial latency
        if response.latency_ms > 0:
            await asyncio.sleep(response.latency_ms / 1000.0)

        for chunk, token_text in self._generate_stream_chunks(response):
            if run_manager and token_text:
                await run_manager.on_llm_new_token(token_text)
            yield chunk
            if response.chunk_delay_ms > 0:
                await asyncio.sleep(response.chunk_delay_ms / 1000.0)
