from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
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
        response = self._perform_api_request(messages, **kwargs)
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
        messages: List[BaseMessage],
        user: StarletteUser,
        role_arn: str,
        **kwargs,
    ):
        q_client = self.amazon_q_client_factory.get_client(
            current_user=user,
            role_arn=role_arn,
        )

        message = "\n".join(
            message.content for message in messages if isinstance(message.content, str)
        )

        return q_client.send_message(message=message)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": "amazon_q",
        }

    @property
    def _llm_type(self) -> str:
        return "amazon_q"
