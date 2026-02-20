import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, MutableMapping, Optional, Sequence, cast

import structlog
from gitlab_cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)
from jinja2 import PackageLoader, meta
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, string
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_core.tools import BaseTool
from langsmith import tracing_context

from ai_gateway.config import ConfigModelLimits, ModelLimits
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.model_metadata import TypeModelMetadata, create_model_metadata
from ai_gateway.model_selection import ModelSelectionConfig, PromptParams
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.bind_tools_cache import BindToolsCacheProtocol
from ai_gateway.prompts.caching import (
    CACHE_CONTROL_INJECTION_POINTS_KEY,
    CacheControlInjectionPointsConverter,
    filter_cache_control_injection_points,
)
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.typing import Model, TypeModelFactory, TypePromptTemplateFactory
from ai_gateway.structured_logging import get_request_logger
from lib.context import StarletteUser, current_model_metadata_context
from lib.internal_events.client import InternalEventsClient

__all__ = [
    "Prompt",
    "BasePromptRegistry",
    "BasePromptCallbackHandler",
    "jinja2_formatter",
    "prompt_template_to_messages",
]


# cspell:ignore binops, binop, Sandboxed
class PromptSandboxedEnvironment(ImmutableSandboxedEnvironment):
    """Sandboxed environment that forbids method access on bound objects."""

    intercepted_binops = frozenset(["*", "**"])

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        if callable(value):
            log.warning(
                "Blocked callable attribute access in prompt template",
                object_type=type(obj).__name__,
                attribute=attr,
            )
            return False
        return super().is_safe_attribute(obj, attr, value)

    def is_safe_callable(self, obj: Any) -> bool:
        return False

    def call_binop(self, context: Any, operator: str, left: Any, right: Any) -> Any:
        if operator in self.intercepted_binops:
            raise SecurityError(
                "Multiplication operators are not allowed in prompt templates"
            )
        return super().call_binop(context, operator, left, right)


def split_filter(
    value: Any,
    delimiter: str = " ",
    maxsplit: Optional[int] = None,
) -> list[str]:
    """Safe string split filter exposed to prompt templates."""

    if delimiter == "":
        raise SecurityError("Empty delimiter is not allowed for split filter")

    split_limit = maxsplit if maxsplit is not None else -1

    return str(value).split(delimiter, split_limit)


jinja_loader = PackageLoader("ai_gateway.prompts", "definitions")
jinja_env = PromptSandboxedEnvironment(loader=jinja_loader)
jinja_env.filters["split"] = split_filter

log = structlog.stdlib.get_logger("prompts")


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


def prompt_template_to_messages(
    tpl: dict[str, str],
) -> Sequence[MessageLikeRepresentation]:
    return [
        MessagesPlaceholder(content) if role == "placeholder" else (role, content)
        for role, content in tpl.items()
    ]


class PromptLoggingHandler(BaseCallbackHandler):
    """Logs the full prompt that is sent to the LLM."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **_kwargs: Any,
    ) -> Any:
        get_request_logger("prompt").info(
            "Performing LLM request", prompt="\n".join(prompts)
        )


class BasePromptCallbackHandler:
    """Base class for defining custom prompt callbacks without LangChain dependencies.

    This class provides a LangChain-agnostic interface for implementing callbacks.
    Developers should extend this class to define custom callbacks outside of the prompt package,
    avoiding the need to spread LangChain dependencies across the codebase.

    Example:
        class MyCustomCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                # Custom logic before LLM call
                print("About to call LLM")

        # Register with a prompt
        prompt.register_internal_callbacks([MyCustomCallback()])
    """

    async def on_before_llm_call(self):
        """Called before an LLM request is made.

        Override this method to implement custom logic that should execute before the LLM is invoked (e.g., logging,
        metrics, billing tracking).
        """


class Prompt(RunnableBinding[Any, BaseMessage]):
    name: str
    model_provider: str
    model: Model
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_tpl: Runnable[Any, PromptValue]
    internal_event_client: Optional[InternalEventsClient] = None
    limits: Optional[ModelLimits] = None
    internal_event_extra: dict[str, Any] = {}
    internal_callbacks: list[BasePromptCallbackHandler] = []

    def __init__(
        self,
        model_provider: ModelClassProvider,
        model_factory: TypeModelFactory,
        config: PromptConfig,
        model_metadata: Optional[TypeModelMetadata] = None,
        prompt_template_factory: Optional[TypePromptTemplateFactory] = None,
        disable_streaming: bool = False,
        tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,
        bind_tools_cache: Optional[BindToolsCacheProtocol] = None,
        bind_tools_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        model_kwargs = self._build_model_kwargs(config.params, model_metadata)
        model = self._build_model(
            model_factory, config.model, model_metadata, disable_streaming
        )

        if tools and isinstance(model, BaseChatModel):
            if bind_tools_cache:
                # Use cached bind_tools to avoid expensive repeated operations
                model_id = f"{model_provider}:{config.model.params.model}"
                model = bind_tools_cache.get_or_bind(  # type: ignore[assignment]
                    model=model,
                    model_id=model_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    model_provider=model_provider,
                    **(bind_tools_params or {}),
                )
            else:
                model = model.bind_tools(  # type: ignore[assignment]
                    tools,
                    tool_choice=tool_choice,
                    **(bind_tools_params or {}),
                )

        prompt = (
            prompt_template_factory(model_provider, config)
            if prompt_template_factory
            else self._build_prompt_template(config)
        )
        prompt = self._chain_cache_control_injection_points_converter(
            model_kwargs, prompt, model_provider
        )

        chain = cast(
            Runnable[Any, BaseMessage],
            prompt
            | model.bind(**model_kwargs).with_config(
                callbacks=[PromptLoggingHandler()]
            ),
        )

        super().__init__(
            name=config.name,
            model_provider=model_provider,
            model=model,
            unit_primitives=config.unit_primitives,
            bound=chain,
            prompt_tpl=prompt,
            **kwargs,
        )  # type: ignore[call-arg]

    def _chain_cache_control_injection_points_converter(
        self,
        model_kwargs: MutableMapping[str, Any],
        prompt: Runnable,
        model_provider: ModelClassProvider,
    ) -> Runnable[Any, PromptValue]:
        """Convert `cache_control_injection_points` LiteLLM param for non-LiteLLM model clients.

        https://docs.litellm.ai/docs/tutorials/prompt_caching
        """

        if (
            CACHE_CONTROL_INJECTION_POINTS_KEY not in model_kwargs
            or model_provider == ModelClassProvider.LITE_LLM
        ):
            return prompt

        chain = prompt | CacheControlInjectionPointsConverter().bind(
            model_class_provider=model_provider,
            cache_control_injection_points=model_kwargs.pop(
                CACHE_CONTROL_INJECTION_POINTS_KEY
            ),
        )

        return chain

    def _build_model_kwargs(
        self,
        params: PromptParams | None,
        model_metadata: Optional[TypeModelMetadata],
    ) -> MutableMapping[str, Any]:
        model_kwargs = {
            **(params.model_dump(exclude_none=True) if params else {}),
            **(model_metadata.to_params() if model_metadata else {}),
        }

        filter_cache_control_injection_points(model_kwargs)

        return model_kwargs

    def _build_model(
        self,
        model_factory: TypeModelFactory,
        config: ModelConfig,
        model_metadata: Optional[TypeModelMetadata],
        disable_streaming: bool,
    ) -> Model:
        # The params in the prompt file have higher precedence than the ones in the model definition
        llm_params = (
            model_metadata.llm_definition.params.model_dump(exclude_none=True)
            if model_metadata
            else {}
        )

        model_factory_args = {
            "disable_streaming": disable_streaming,
            **llm_params,
            **config.params.model_dump(exclude_none=True, by_alias=True),
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
            model_provider=self.model_provider,
        )

    def set_limits(self, model_limits: ConfigModelLimits):
        self.limits = model_limits.for_model(
            engine=self.model._llm_type, name=self.model_name
        )

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> BaseMessage:
        with (
            self.instrumentator.watch(
                stream=False,
                unit_primitives=self.unit_primitives,
                internal_event_client=self.internal_event_client,
            ) as watcher,
            get_usage_metadata_callback() as cb,
        ):
            await asyncio.gather(
                *[cb.on_before_llm_call() for cb in self.internal_callbacks]
            )

            result = await super().ainvoke(
                input,
                config,
                **kwargs,
            )

            watcher.register_message(result)
            self.handle_usage_metadata(watcher, cb.usage_metadata)

            return result

    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[BaseMessage]:
        # pylint: disable=contextmanager-generator-missing-cleanup,line-too-long
        # To properly address this pylint issue, the upstream function would need to be altered to ensure proper cleanup.
        # See https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/contextmanager-generator-missing-cleanup.html
        with (
            self.instrumentator.watch(
                stream=True,
                unit_primitives=self.unit_primitives,
                internal_event_client=self.internal_event_client,
            ) as watcher,
            get_usage_metadata_callback() as cb,
        ):
            # The usage metadata callback only totals the usage at the `on_llm_end` event, so we need to be able to
            # yield the last stream item _after_ that event. Otherwise we'd need to yield an extra event just for the
            # usage metadata. To do this, we yield with a 1-item offset.
            previous_item: BaseMessage | None = None

            await asyncio.gather(
                *[cb.on_before_llm_call() for cb in self.internal_callbacks]
            )

            async for item in super().astream(
                input,
                config,
                **kwargs,
            ):
                watcher.register_message(item)

                if previous_item:
                    yield previous_item
                previous_item = item

            self.handle_usage_metadata(watcher, cb.usage_metadata)

            # Now the usage metadata is available
            if previous_item:
                yield previous_item

            await watcher.afinish()
        # pylint: enable=contextmanager-generator-missing-cleanup,line-too-long

    async def atransform(
        self,
        input: AsyncIterator[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[BaseMessage]:
        # See note in `astream` about the cleanup pylint suppression
        with (  # pylint: disable=contextmanager-generator-missing-cleanup
            self.instrumentator.watch(
                stream=True,
                unit_primitives=self.unit_primitives,
                internal_event_client=self.internal_event_client,
            ) as watcher,
            get_usage_metadata_callback() as cb,
        ):
            # See `astream` comments for why we yield off by one
            previous_item: BaseMessage | None = None

            async for item in super().atransform(input, config, **kwargs):
                watcher.register_message(item)

                if previous_item:
                    yield previous_item
                previous_item = item

            self.handle_usage_metadata(watcher, cb.usage_metadata)

            if previous_item:
                yield previous_item

            await watcher.afinish()

    def handle_usage_metadata(
        self,
        watcher: ModelRequestInstrumentator.WatchContainer,
        usage_metadata: dict[str, UsageMetadata],
    ) -> None:
        for model, usage in usage_metadata.items():
            watcher.register_token_usage(model, usage, self.internal_event_extra)

    @classmethod
    def _build_prompt_template(cls, config: PromptConfig) -> Runnable[Any, PromptValue]:
        messages = prompt_template_to_messages(config.prompt_template)

        return cast(
            Runnable[Any, PromptValue],
            ChatPromptTemplate.from_messages(messages, template_format="jinja2"),
        )


class BasePromptRegistry(ABC):
    internal_event_client: InternalEventsClient
    model_limits: ConfigModelLimits
    _DEFAULT_VERSION: str | None = "^1.0.0"
    validations: set[str] | None = None

    @abstractmethod
    def get(
        self,
        prompt_id: str,
        prompt_version: str | None,
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
            prompt_id,
            prompt_version or self._DEFAULT_VERSION,
            model_metadata,
            tools,
            **kwargs,
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

    async def validate_model(self, model: str):
        log.info("Validating default model", model=model)

        prompt = self.get(
            "model_configuration/check",
            self._DEFAULT_VERSION,
            model_metadata=create_model_metadata({"provider": "gitlab", "name": model}),
        )

        await prompt.ainvoke({})

        if self.validations:
            # Persist validations so we don't incur in multiple 3rd party LLM calls from multiple invocations
            self.validations.add(model)

    async def validate_default_models(
        self, unit_primitive: GitLabUnitPrimitive | None = None
    ) -> bool:
        model_selection_config = ModelSelectionConfig.instance()

        if self.validations is None:
            # TODO: Remove this exception once the prompt registry properly supports Fireworks.
            # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/906
            self.validations = set({"codestral_2501_fireworks"})

        # Collect invocations to execute them in parallel
        tasks = []

        for unit_primitive_config in model_selection_config.get_unit_primitive_config():
            model = unit_primitive_config.default_model

            if model in self.validations or (
                unit_primitive
                and unit_primitive not in unit_primitive_config.unit_primitives
            ):
                continue

            tasks.append(self.validate_model(model))

        with tracing_context(enabled=False):
            await asyncio.gather(*tasks)

        return True
