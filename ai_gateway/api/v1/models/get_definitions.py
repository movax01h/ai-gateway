from typing import Literal

from fastapi import APIRouter, status
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel
from starlette.responses import JSONResponse

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.model_selection.types import DeprecationInfo, DevConfig
from lib.feature_flags import FeatureFlag, is_feature_enabled

router = APIRouter()

# Identifier used for the pseudo-model created when a unit primitive has multiple default models.
# The pseudo-model represents the load-balanced default and has no provider suffix in its name,
# since requests are distributed across providers transparently.
# NOTE: Multiple default models should only be used for the same model across different providers,
# because a single model name (from the first default model) will be used when displaying the
# default model to the user.
_PSEUDO_DEFAULT_MODEL_IDENTIFIER = "__default__"


class _GetModelResponseModel(BaseModel):
    name: str
    identifier: str
    provider: str | None = None
    deprecation: DeprecationInfo | None = None
    description: str | None = None
    cost_indicator: str | None = None


class _GetModelResponseUnitPrimitive(BaseModel):
    feature_setting: str
    default_model: str  # deprecated, maintained for backward compatibility
    default_models: list[str]
    models_for_size_preference: dict[Literal["small", "large"], str]
    selectable_models: list[str]
    beta_models: list[str]
    dev: DevConfig | None
    unit_primitives: list[GitLabUnitPrimitive]


class _GetModelResponse(BaseModel):
    models: list[_GetModelResponseModel]
    unit_primitives: list[_GetModelResponseUnitPrimitive]


@router.get(
    "/definitions",
    status_code=status.HTTP_200_OK,
    description="List of available large language models powering GitLab Duo features",
)
async def get_models():
    selection_config = ModelSelectionConfig.instance()
    llm_definitions = selection_config.get_llm_definitions()
    unit_primitives = []
    # Collect pseudo-models to add for unit primitives with multiple default models.
    # Maps pseudo-model identifier to _GetModelResponseModel.
    pseudo_models: dict[str, _GetModelResponseModel] = {}

    multi_default_models_enabled = is_feature_enabled(
        FeatureFlag.AI_GATEWAY_MULTI_DEFAULT_MODELS
    )

    for primitive in selection_config.get_unit_primitive_config():
        values = primitive.model_dump()
        default_models = values["default_models"]

        if len(default_models) > 1 and multi_default_models_enabled:
            # When there are multiple default models (load balancing across providers),
            # create a pseudo-model using the first model's name without a provider suffix.
            # This pseudo-model is used as the displayed default in the UI.
            first_model_id = default_models[0]
            first_definition = llm_definitions.get(first_model_id)
            pseudo_name = first_definition.name if first_definition else first_model_id
            # Use a feature-setting-scoped identifier for the pseudo-model
            pseudo_identifier = (
                f"{_PSEUDO_DEFAULT_MODEL_IDENTIFIER}{primitive.feature_setting}"
            )
            pseudo_models[pseudo_identifier] = _GetModelResponseModel(
                name=pseudo_name,
                identifier=pseudo_identifier,
                provider=None,
                description=(
                    first_definition.description if first_definition else None
                ),
                cost_indicator=(
                    first_definition.cost_indicator if first_definition else None
                ),
            )
            values["default_model"] = pseudo_identifier
            # Surface the pseudo-model in the dropdown as the prominent default row;
            # per-provider variants stay in the list so users can still pin one.
            values["selectable_models"] = [
                pseudo_identifier,
                *values["selectable_models"],
            ]
        else:
            values["default_model"] = default_models[0]

        unit_primitives.append(_GetModelResponseUnitPrimitive(**values))

    # Build the models list: append " - [provider]" to names of models that have a provider.
    models = [
        _GetModelResponseModel(
            name=(
                f"{definition.name} - {definition.provider}"
                if definition.provider
                else definition.name
            ),
            identifier=definition.gitlab_identifier,
            provider=definition.provider,
            deprecation=definition.deprecation,
            description=definition.description,
            cost_indicator=definition.cost_indicator,
        )
        for definition in llm_definitions.values()
    ]

    # Append pseudo-models to the models list so clients can resolve the default_model identifier.
    models.extend(pseudo_models.values())

    response = _GetModelResponse(
        models=models,
        unit_primitives=unit_primitives,
    )

    return JSONResponse(content=response.model_dump(mode="json"))
