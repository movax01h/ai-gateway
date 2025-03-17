from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory

__all__ = [
    "ChatAmazonQ",
]


class ChatAmazonQ(BaseChatModel):
    amazon_q_client_factory: AmazonQClientFactory

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        generations = [ChatGeneration(message=AIMessage(content="Amazon Q"))]

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

        finally:
            stream.close()

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
        system_message = messages.pop(0)
        assistant_message = messages.pop()
        user_message = messages.pop()

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"userInputMessage": {"content": str(msg.content)}})
            elif isinstance(msg, AIMessage):
                history.append(
                    {"assistantResponseMessage": {"content": str(msg.content)}}
                )

        message = {
            "content": f"{system_message.content} {user_message.content} {assistant_message.content}"
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
