import functools
import typing

from fastapi import HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive
from gitlab_cloud_connector.data_model import CloudConnectorCatalog, load_catalog
from starlette_context import context

X_GITLAB_UNIT_PRIMITIVE = "x-gitlab-unit-primitive"

_CATEGORY_CONTEXT_KEY = "meta.feature_category"
_UNIT_PRIMITIVE_CONTEXT_KEY = "meta.unit_primitive"
_UNKNOWN_FEATURE_CATEGORY = "unknown"

# Attribute set on decorated endpoints so the declared categories can be inspected without a request.
# `functools.wraps` copies it to any decorator applied on top.
_FEATURE_CATEGORIES_ATTR = "_gitlab_feature_categories"

# Loaded once per process, like `_config` in `ai_gateway/config.py`: `load_catalog` parses every unit primitive
# definition on each call.
_catalog = load_catalog()


def get_catalog() -> CloudConnectorCatalog:
    """Get the Cloud Connector catalog, loaded once per process."""
    return _catalog


def feature_category(name: GitLabFeatureCategory):
    """Track a feature category in a single purpose endpoint.

    Example:

    ```
    @feature_category(GitLabFeatureCategory.DUO_CHAT)
    ```
    """
    try:
        GitLabFeatureCategory(name)
    except ValueError:
        raise ValueError(f"Invalid feature category: {name}")

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            context[_CATEGORY_CONTEXT_KEY] = name
            return await func(*args, **kwargs)

        setattr(wrapper, _FEATURE_CATEGORIES_ATTR, frozenset({name}))
        return wrapper

    return decorator


def feature_categories(mapping: dict[GitLabUnitPrimitive, GitLabFeatureCategory]):
    """Track feature categories in a multi purpose endpoint.

    It gets the purpose of API call from X-GitLab-Unit-Primitive header,
    identifies the corresponding feature category and stores them in the Starlette context.

    Example:

    ```
    @feature_categories({
        GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
        GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    })
    ```
    """
    for category in mapping.values():
        try:
            GitLabFeatureCategory(category)
        except ValueError:
            raise ValueError(f"Invalid feature category: {category}")

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            try:
                unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing {X_GITLAB_UNIT_PRIMITIVE} header",
                )

            try:
                gitlab_feature_category = mapping[unit_primitive]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This endpoint cannot be used for {unit_primitive} purpose",
                )

            context[_CATEGORY_CONTEXT_KEY] = gitlab_feature_category
            context[_UNIT_PRIMITIVE_CONTEXT_KEY] = unit_primitive
            return await func(request, *args, **kwargs)

        setattr(wrapper, _FEATURE_CATEGORIES_ATTR, frozenset(mapping.values()))
        return wrapper

    return decorator


def set_feature_category_for_unit_primitive(
    unit_primitive: GitLabUnitPrimitive,
) -> None:
    """Record the feature category that owns a unit primitive in the request context.

    The category comes from the Cloud Connector catalog, so it may not be a `GitLabFeatureCategory` member. If the
    catalog has no entry for the unit primitive, the current category is left unchanged.
    """
    for definition in get_catalog().unit_primitives:
        if definition.name == unit_primitive:
            context[_CATEGORY_CONTEXT_KEY] = definition.feature_category
            return


def get_feature_categories(
    endpoint: typing.Callable,
) -> frozenset[GitLabFeatureCategory]:
    """Get the feature categories declared on an endpoint with `feature_category` or `feature_categories`.

    Returns an empty set if the endpoint declares none.
    """
    return getattr(endpoint, _FEATURE_CATEGORIES_ATTR, frozenset())


def current_feature_category() -> str:
    """Get the feature category set to the current request context."""
    if context.exists():
        gitlab_feature_category = context.get(
            _CATEGORY_CONTEXT_KEY, _UNKNOWN_FEATURE_CATEGORY
        )

        if isinstance(gitlab_feature_category, GitLabFeatureCategory):
            return gitlab_feature_category.value

        return gitlab_feature_category

    return _UNKNOWN_FEATURE_CATEGORY
