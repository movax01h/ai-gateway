from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory

__all__ = [
    "ChatAmazonQ",
]


class ReferenceSpan(BaseModel):
    shape: str


class Reference(BaseModel):
    repository: Optional[ReferenceSpan | str] = None
    licenseName: Optional[ReferenceSpan | str] = None
    url: Optional[ReferenceSpan | str] = None
    recommendationContentSpan: Optional[ReferenceSpan | str] = None

    def get_repository(self) -> Optional[str]:
        return (
            self.repository.shape
            if isinstance(self.repository, ReferenceSpan)
            else self.repository
        )

    def get_license_name(self) -> Optional[str]:
        return (
            self.licenseName.shape
            if isinstance(self.licenseName, ReferenceSpan)
            else self.licenseName
        )

    def get_url(self) -> Optional[str]:
        return self.url.shape if isinstance(self.url, ReferenceSpan) else self.url

    def get_span(self) -> Optional[str]:
        return (
            self.recommendationContentSpan.shape
            if isinstance(self.recommendationContentSpan, ReferenceSpan)
            else self.recommendationContentSpan
        )

    def format_reference(self) -> str:
        parts = []

        if repository := self.get_repository():
            parts.append(str(repository))
        if license_name := self.get_license_name():
            parts.append(f"[{license_name}]")
        if url := self.get_url():
            parts.append(f": {url}")
        if span := self.get_span():
            parts.append(f"({span})")

        return " ".join(parts)


class ChatAmazonQ(BaseChatModel):
    amazon_q_client_factory: AmazonQClientFactory

    def _generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResult:
        content = "".join(
            chunk.message.content
            for chunk in self._stream(*args, **kwargs)
            if isinstance(chunk.message.content, str)
        )

        generations = [ChatGeneration(message=AIMessage(content=content))]

        return ChatResult(generations=generations)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message, history = self._build_messages(messages)
        response = self._perform_api_request(message, history, **kwargs)
        stream = response["responseStream"]

        try:
            for event in stream:
                for key, value in event.items():
                    if key == "assistantResponseEvent":
                        content = value.get("content")
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=content)
                        )
                    elif key == "codeReferenceEvent":
                        yield from self._process_code_reference_event(event)

        finally:
            stream.close()

    def _process_code_reference_event(
        self, event: Dict
    ) -> Iterator[ChatGenerationChunk]:
        """
        Process code reference events and format them into a readable string.
        Uses Pydantic models for data validation and parsing.

        Args:
            event (Dict): The event containing code references

        Returns:
            Iterator[ChatGenerationChunk]: A response containing the code reference information

        Example:
            Input event:
            {
                "codeReferenceEvent": {
                    "references": [
                        {
                            "repository": {"shape": "example-repo"},
                            "licenseName": {"shape": "MIT"},
                            "url": {"shape": "https://github.com/example/repo"},
                            "recommendationContentSpan": {"shape": "lines 10-20"}
                        },
                        {
                            "repository": {"shape": "another-repo"},
                            "licenseName": {"shape": "Apache-2.0"},
                            "url": {"shape": "https://github.com/another/repo"},
                            "recommendationContentSpan": {"shape": "lines 5-15"}
                        }
                    ]
                }
            }

        Output:
            ChatGenerationChunk with content:
            "example-repo [MIT]: https://github.com/example/repo (lines 10-20)
            another-repo [Apache-2.0]: https://github.com/another/repo (lines 5-15)"
        """
        try:
            references = event.get("codeReferenceEvent", {}).get("references", [])
            formatted_references = []

            for reference in references:
                try:
                    # Using model_validate instead of parse_obj
                    ref = Reference.model_validate(reference)
                    formatted_ref = ref.format_reference()
                    if formatted_ref:
                        formatted_references.append(formatted_ref)
                except ValueError:
                    # Handle validation errors if needed
                    continue

            if formatted_references:
                reference_content = "\n".join(formatted_references)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=reference_content)
                )
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))

        except Exception:
            # If there's any error in parsing/validation, return empty content
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    def _perform_api_request(
        self,
        message: dict[str, str],
        history: List[dict[str, str]],
        user: StarletteUser,
        role_arn: str,
        **_kwargs,
    ):
        """
        Performs a `send_message` request to Q API.

        This method creates a Q client and performs a `send_message` request passing `message` and `history`.

        Args:
            message (dict): A dictionary with a "content" key that combines the system and the latest user and assistant messages.
            history (list): A list of dictionaries representing user and assistant message history,
                            with either {"userInputMessage": { "content" ... }} or {"assistantResponseMessage": {"content" ... }} formats.
            user (StarletteUser): The current user who performs the request.
            role_arn (str): The role arn of the identity provider.
            kwargs (dict): Optional arguments.

        Returns:
            dict: A dict with "responseStream" key that contains a stream of events.
        """
        q_client = self.amazon_q_client_factory.get_client(
            current_user=user,
            role_arn=role_arn,
        )

        return q_client.send_message(message=message, history=history)

    def _build_messages(
        self,
        messages: List[BaseMessage],
    ):
        """
        Build a message and history from a list of provided messages that can be later passed to the `send_message` endpoint of Q API.

        Args:
            messages (List[BaseMessage]): A list of messages, including system, user, and assistant messages.

        Returns:
            tuple: A tuple containing:
                - message (dict): A dictionary with a "content" key that combines the system and the latest
                  user and assistant messages.
                - history (list): A list of dictionaries representing user and assistant message history,
                  with either {"userInputMessage": { "content" ... }} or {"assistantResponseMessage": {"content" ... }} formats.
        """
        input_messages = []
        # Extract the system message to always send it as an input
        if messages and isinstance(messages[0], SystemMessage):
            input_messages.append(messages.pop(0))
        # Support prompt definitions with assistant messages (like react prompts)
        if len(messages) > 1 and isinstance(messages[-1], AIMessage):
            assistant_message = messages.pop()
            user_message = messages.pop()
            input_messages.append(user_message)
            input_messages.append(assistant_message)
        # Support prompt definitions with system + user messages (like explain code prompts)
        if messages and isinstance(messages[-1], HumanMessage):
            input_messages.append(messages.pop())

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"userInputMessage": {"content": str(msg.content)}})
            elif isinstance(msg, AIMessage):
                history.append(
                    {"assistantResponseMessage": {"content": str(msg.content)}}
                )

        message = {
            "content": " ".join(
                msg.content for msg in input_messages if isinstance(msg.content, str)
            )
        }

        return message, history

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": "amazon_q",
        }

    @property
    def _llm_type(self) -> str:
        return "amazon_q"
