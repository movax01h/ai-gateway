from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Mapping, Optional, Sequence, TypeVar, cast

from gitlab_cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)
from jinja2 import PackageLoader, meta
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, string
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_core.tools import BaseTool

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits, ModelLimits
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.model_metadata import TypeModelMetadata, current_model_metadata_context
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.typing import Model, TypeModelFactory
from ai_gateway.structured_logging import get_request_logger
from duo_workflow_service.tracking.llm_usage_context import get_workflow_checkpointer
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties

__all__ = [
    "Prompt",
    "Input",
    "Output",
    "BasePromptRegistry",
    "jinja2_formatter",
]

Input = TypeVar("Input")
Output = TypeVar("Output")

jinja_loader = PackageLoader("ai_gateway.prompts", "definitions")
jinja_env = SandboxedEnvironment(loader=jinja_loader)


def _get_jinja2_variables_from_template(template: str) -> set[str]:
    ast = jinja_env.parse(template)
    variables = meta.find_undeclared_variables(ast)

    for template_name in meta.find_referenced_templates(ast):
        if not template_name:
            continue

        template_source, _, _ = jinja_loader.get_source(jinja_env, template_name)
        ast = jinja_env.parse(template_source)
        variables = variables.union(meta.find_undeclared_variables(ast))

    return variables


string._get_jinja2_variables_from_template = _get_jinja2_variables_from_template


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    return jinja_env.from_string(template).render(**kwargs)


# Override LangChain's jinja2 formatter so we can specify a loader with access to all our templates
DEFAULT_FORMATTER_MAPPING["jinja2"] = jinja2_formatter


class PromptLoggingHandler(BaseCallbackHandler):
    """Logs the full prompt that is sent to the LLM."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],  # pylint: disable=unused-argument
        prompts: list[str],
        **_kwargs: Any,
    ) -> Any:
        get_request_logger("prompt").info(
            "Performing LLM request", prompt="\n".join(prompts)
        )


class Prompt(RunnableBinding[Input, Output]):
    name: str
    model_engine: str
    model_provider: str
    model: Model
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_tpl: Runnable[Input, PromptValue]
    internal_event_client: Optional[InternalEventsClient] = None
    limits: Optional[ModelLimits] = None

    def __init__(
        self,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[TypeModelMetadata] = None,
        disable_streaming: bool = False,
        tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ):
        model_provider = config.model.params.model_class_provider
        model_kwargs = self._build_model_kwargs(config.params, model_metadata)
        model = self._build_model(
            model_factory, config.model, model_metadata, disable_streaming
        )

        if tools and isinstance(model, BaseChatModel):
            model = model.bind_tools(tools, tool_choice=tool_choice)  # type: ignore[assignment]

        prompt = self._build_prompt_template(config)
        chain = self._build_chain(
            cast(
                Runnable[Input, Output],
                prompt | model.bind(**model_kwargs),
            )
        )

        super().__init__(
            name=config.name,
            model_engine=config.model.params.custom_llm_provider or model_provider,
            model_provider=model_provider,
            model=model,
            unit_primitives=config.unit_primitives,
            bound=chain,
            prompt_tpl=prompt,
            **kwargs,
        )  # type: ignore[call-arg]

    def _build_model_kwargs(
        self,
        params: PromptParams | None,
        model_metadata: Optional[TypeModelMetadata],
    ) -> Mapping[str, Any]:
        return {
            **(params.model_dump(exclude_none=True) if params else {}),
            **(model_metadata.to_params() if model_metadata else {}),
        }

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
        model_metadata: Optional[TypeModelMetadata],
        disable_streaming: bool,
    ) -> Model:
        # The params in the prompt file have higher precedence than the ones in the model definition
        model_factory_args = {
            "disable_streaming": disable_streaming,
            **(model_metadata.llm_definition_params if model_metadata else {}),
            **config.params.model_dump(
                exclude={"model_class_provider"}, exclude_none=True, by_alias=True
            ),
        }
        return model_factory(**model_factory_args)

    @property
    def model_name(self) -> str:
        return self.model._identifying_params["model"]

    @property
    def instrumentator(self) -> ModelRequestInstrumentator:
        return ModelRequestInstrumentator(
            model_engine=self.model._llm_type,
            model_name=self.model_name,
            limits=self.limits,
        )

    @property
    def internal_event_extra(self) -> dict[str, Any]:
        return {}

    def set_limits(self, model_limits: ConfigModelLimits):
        self.limits = model_limits.for_model(
            engine=self.model._llm_type, name=self.model_name
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        with (
            self.instrumentator.watch(
                stream=False, unit_primitives=self.unit_primitives
            ) as watcher,
            get_usage_metadata_callback() as cb,
        ):
            result = await super().ainvoke(
                input,
                self._add_logger_to_config(config),
                **kwargs,
            )

            self.handle_usage_metadata(watcher, cb.usage_metadata)

            return result

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        # pylint: disable=contextmanager-generator-missing-cleanup,line-too-long
        # To properly address this pylint issue, the upstream function would need to be altered to ensure proper cleanup.
        # See https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/contextmanager-generator-missing-cleanup.html
        with (
            self.instrumentator.watch(
                stream=True, unit_primitives=self.unit_primitives
            ) as watcher,
            get_usage_metadata_callback() as cb,
        ):
            # The usage metadata callback only totals the usage at the `on_llm_end` event, so we need to be able to
            # yield the last stream item _after_ that event. Otherwise we'd need to yield an extra event just for the
            # usage metadata. To do this, we yield with a 1-item offset.
            previous_item: Output | None = None

            async for item in super().astream(
                input,
                self._add_logger_to_config(config),
                **kwargs,
            ):
                if previous_item:
                    yield previous_item
                previous_item = item

            self.handle_usage_metadata(watcher, cb.usage_metadata)

            # Now the usage metadata is available
            if previous_item:
                yield previous_item

            await watcher.afinish()
        # pylint: enable=contextmanager-generator-missing-cleanup,line-too-long

    def handle_usage_metadata(
        self,
        watcher: ModelRequestInstrumentator.WatchContainer,
        usage_metadata: dict[str, UsageMetadata],
    ) -> None:

        get_request_logger("prompt").info(
            f"LLM call finished with token usage: {usage_metadata}"
        )
        checkpointer = get_workflow_checkpointer()
        if self.internal_event_client is None and checkpointer is None:
            return

        for model, usage in usage_metadata.items():
            watcher.register_token_usage(model, usage)
            if checkpointer:
                checkpointer.track_llm_operation(
                    token_count=usage["total_tokens"],
                    model_id=model,
                    model_engine=self.model_engine,
                    model_provider=self.model_provider,
                    prompt_tokens=usage["input_tokens"],
                    completion_tokens=usage["output_tokens"],
                )

            for unit_primitive in self.unit_primitives:
                # Access langchain usage_metadata for optional cache
                # specific token details
                input_token_details = usage.get("input_token_details", {})
                cache_creation = input_token_details.get("cache_creation", 0)
                cache_read = input_token_details.get("cache_read", 0)

                # Optional event tracking for TTL prompt caching
                ephemeral_5m_input_tokens = input_token_details.get(
                    "ephemeral_5m_input_tokens", 0
                )
                ephemeral_1h_input_tokens = input_token_details.get(
                    "ephemeral_1h_input_tokens", 0
                )

                additional_properties = InternalEventAdditionalProperties(
                    label="cache_details",
                    cache_read=cache_read,
                    cache_creation=cache_creation,
                    ephemeral_5m_input_tokens=ephemeral_5m_input_tokens,
                    ephemeral_1h_input_tokens=ephemeral_1h_input_tokens,
                    **self.internal_event_extra,
                )

                if self.internal_event_client:
                    self.internal_event_client.track_event(
                        f"token_usage_{unit_primitive}",
                        category=__name__,
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        total_tokens=usage["total_tokens"],
                        model_engine=self.model_engine,
                        model_name=model,
                        model_provider=self.model_provider,
                        additional_properties=additional_properties,
                    )

    # Subclasses can override this method to add steps at either side of the chain
    @staticmethod
    def _build_chain(chain: Runnable[Input, Output]) -> Runnable[Input, Output]:
        return chain

    @staticmethod
    def _add_logger_to_config(config):
        callback = PromptLoggingHandler()

        if not config:
            return {"callbacks": [callback]}

        config["callbacks"] = [*config.get("callbacks", []), callback]

        return config

    # Assume that the prompt template keys map to roles. Subclasses can
    # override this method to implement more complex logic.
    @staticmethod
    def _prompt_template_to_messages(
        tpl: dict[str, str],
    ) -> Sequence[MessageLikeRepresentation]:
        return [
            MessagesPlaceholder(content) if role == "placeholder" else (role, content)
            for role, content in tpl.items()
        ]

    @classmethod
    def _build_prompt_template(
        cls, config: PromptConfig
    ) -> Runnable[Input, PromptValue]:
        messages = cls._prompt_template_to_messages(config.prompt_template)

        return cast(
            Runnable[Input, PromptValue],
            ChatPromptTemplate.from_messages(messages, template_format="jinja2"),
        )


class BasePromptRegistry(ABC):
    internal_event_client: InternalEventsClient
    model_limits: ConfigModelLimits

    @abstractmethod
    def get(
        self,
        prompt_id: str,
        prompt_version: str,
        model_metadata: Optional[TypeModelMetadata] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Prompt:
        pass

    def get_on_behalf(
        self,
        # TODO: We should allow only `CloudConnectorUser` in the future.
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1224
        user: StarletteUser | CloudConnectorUser,
        prompt_id: str,
        prompt_version: Optional[str] = None,
        model_metadata: Optional[TypeModelMetadata] = None,
        internal_event_category=__name__,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Prompt:
        if not model_metadata:
            model_metadata = current_model_metadata_context.get()

        if model_metadata and isinstance(user, StarletteUser):
            model_metadata.add_user(user)

        prompt = self.get(
            prompt_id, prompt_version or "^1.0.0", model_metadata, tools, **kwargs
        )
        prompt.internal_event_client = self.internal_event_client
        prompt.set_limits(self.model_limits)

        for unit_primitive in prompt.unit_primitives:
            if not user.can(unit_primitive):
                raise WrongUnitPrimitives

        # Only record internal events once we know the user has access to all Unit Primitives
        for unit_primitive in prompt.unit_primitives:
            self.internal_event_client.track_event(
                f"request_{unit_primitive}", category=internal_event_category
            )

        return prompt
