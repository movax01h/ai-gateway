from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Mapping, Optional, Tuple, TypeVar, cast

from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from jinja2 import PackageLoader
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.internal_events.client import InternalEventsClient
from ai_gateway.model_metadata import TypeModelMetadata, current_model_metadata_context
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.typing import Model, TypeModelFactory

__all__ = [
    "Prompt",
    "BasePromptRegistry",
    "jinja2_formatter",
]

Input = TypeVar("Input")
Output = TypeVar("Output")

jinja_env = SandboxedEnvironment(
    loader=PackageLoader("ai_gateway.prompts", "definitions")
)


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    return jinja_env.from_string(template).render(**kwargs)


# Override LangChain's jinja2 formatter so we can specify a loader with access to all our templates
DEFAULT_FORMATTER_MAPPING["jinja2"] = jinja2_formatter


class Prompt(RunnableBinding[Input, Output]):
    name: str
    model: Model
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_tpl: Runnable[Input, PromptValue]

    def __init__(
        self,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[TypeModelMetadata] = None,
        disable_streaming: bool = False,
    ):
        model_override = None

        model_kwargs = self._build_model_kwargs(config.params, model_metadata)
        model = self._build_model(
            model_factory, config.model, disable_streaming, model_override
        )
        prompt = self._build_prompt_template(config.prompt_template, config.model)
        chain = self._build_chain(
            cast(Runnable[Input, Output], prompt | model.bind(**model_kwargs))
        )

        super().__init__(
            name=config.name,
            model=model,
            unit_primitives=config.unit_primitives,
            bound=chain,
            prompt_tpl=prompt,
        )  # type: ignore[call-arg]

    def _build_model_kwargs(
        self,
        params: PromptParams | None,
        model_metadata: Optional[TypeModelMetadata] | None,
    ) -> Mapping[str, Any]:
        return {
            **(params.model_dump(exclude_none=True) if params else {}),
            **(model_metadata.to_params() if model_metadata else {}),
        }

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
        disable_streaming: bool,
        model_override: Optional[str] = None,
    ) -> Model:
        return model_factory(
            model=model_override or config.name,
            disable_streaming=disable_streaming,
            **config.params.model_dump(
                exclude={"model_class_provider"}, exclude_none=True, by_alias=True
            ),
        )

    @property
    def model_name(self) -> str:
        return self.model._identifying_params["model"]

    @property
    def instrumentator(self) -> ModelRequestInstrumentator:
        return ModelRequestInstrumentator(
            model_engine=self.model._llm_type,
            model_name=self.model_name,
            concurrency_limit=None,  # TODO: Plug concurrency limit into agents
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        with self.instrumentator.watch(stream=False):
            return await super().ainvoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        with self.instrumentator.watch(stream=True) as watcher:
            async for item in super().astream(input, config, **kwargs):
                yield item

            await watcher.afinish()

    # Subclasses can override this method to add steps at either side of the chain
    @staticmethod
    def _build_chain(chain: Runnable[Input, Output]) -> Runnable[Input, Output]:
        return chain

    # Assume that the prompt template keys map to roles. Subclasses can
    # override this method to implement more complex logic.
    @staticmethod
    def _prompt_template_to_messages(tpl: dict[str, str]) -> list[Tuple[str, str]]:
        return list(tpl.items())

    @classmethod
    def _build_prompt_template(
        cls, prompt_template: dict[str, str], model_config: ModelConfig
    ) -> Runnable[Input, PromptValue]:
        messages = []

        for role, template in cls._prompt_template_to_messages(prompt_template):
            messages.append((role, template))

        return cast(
            Runnable[Input, PromptValue],
            ChatPromptTemplate.from_messages(messages, template_format="jinja2"),
        )


class BasePromptRegistry(ABC):
    internal_event_client: InternalEventsClient

    @abstractmethod
    def get(
        self,
        prompt_id: str,
        prompt_version: str,
        model_metadata: Optional[TypeModelMetadata] = None,
    ) -> Prompt:
        pass

    def get_on_behalf(
        self,
        user: StarletteUser,
        prompt_id: str,
        prompt_version: Optional[str] = None,
        model_metadata: Optional[TypeModelMetadata] = None,
        internal_event_category=__name__,
    ) -> Prompt:
        if not model_metadata:
            model_metadata = current_model_metadata_context.get()

        if model_metadata:
            model_metadata.add_user(user)

        prompt = self.get(prompt_id, prompt_version or "^1.0.0", model_metadata)

        for unit_primitive in prompt.unit_primitives:
            if not user.can(unit_primitive):
                raise WrongUnitPrimitives

        # Only record internal events once we know the user has access to all Unit Primitives
        for unit_primitive in prompt.unit_primitives:
            self.internal_event_client.track_event(
                f"request_{unit_primitive}", category=internal_event_category
            )

        return prompt
