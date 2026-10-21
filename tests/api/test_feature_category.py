# from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
import functools
from enum import StrEnum
from unittest import mock
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive
from starlette_context import request_cycle_context

from ai_gateway.api.feature_category import (
    current_feature_category,
    feature_categories,
    feature_category,
    get_catalog,
    get_feature_categories,
    set_feature_category_for_unit_primitive,
)


class DummyGitLabFeatureCategory(StrEnum):
    AWESOME_CATEGORY_1 = "awesome_category_1"
    AWESOME_CATEGORY_2 = "awesome_category_2"


class DummyGitLabUnitPrimitive(StrEnum):
    AWESOME_FEATURE_1 = "awesome_feature_1"
    AWESOME_FEATURE_2 = "awesome_feature_2"


@pytest.fixture(name="patch_feature_category")
def patch_feature_category_fixture():
    patcher = patch(
        "ai_gateway.api.feature_category.GitLabFeatureCategory",
        spec=GitLabFeatureCategory,
    )
    mock_thing = patcher.start()
    mock_thing.side_effect = DummyGitLabFeatureCategory
    yield
    patcher.stop()


@pytest.fixture(name="patch_unit_primitive")
def patch_unit_primitive_fixture():
    patcher = patch("ai_gateway.api.feature_category.GitLabUnitPrimitive")
    mock_thing = patcher.start()
    mock_thing.side_effect = DummyGitLabUnitPrimitive
    yield
    patcher.stop()


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_feature_category")
async def test_feature_category():
    @feature_category(DummyGitLabFeatureCategory.AWESOME_CATEGORY_1)
    async def to_be_decorated():
        pass

    with patch("ai_gateway.api.feature_category.context") as mock_context:
        await to_be_decorated()

        mock_context.__setitem__.assert_called_once_with(
            "meta.feature_category", DummyGitLabFeatureCategory.AWESOME_CATEGORY_1
        )


def test_unknown_feature_category():
    with pytest.raises(ValueError) as error:
        feature_category("not_exist")

    assert str(error.value) == "Invalid feature category: not_exist"


def test_current_feature_category():
    with request_cycle_context(
        {"meta.feature_category": GitLabFeatureCategory.CODE_SUGGESTIONS}
    ):
        assert current_feature_category() == "code_suggestions"

    with request_cycle_context({"meta.feature_category": "awesome_category"}):
        assert current_feature_category() == "awesome_category"

    # When there value isn't set in the context
    with request_cycle_context({}):
        assert current_feature_category() == "unknown"

    # When the context isn't initialized
    assert current_feature_category() == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected_unit_primitive", "expected_category", "expected_error"),
    [
        (
            {"x-gitlab-unit-primitive": DummyGitLabUnitPrimitive.AWESOME_FEATURE_1},
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_1,
            DummyGitLabFeatureCategory.AWESOME_CATEGORY_1,
            "",
        ),
        (
            {"x-gitlab-unit-primitive": DummyGitLabUnitPrimitive.AWESOME_FEATURE_2},
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_2,
            DummyGitLabFeatureCategory.AWESOME_CATEGORY_2,
            "",
        ),
        ({}, None, None, "400: Missing x-gitlab-unit-primitive header"),
        (
            {"x-gitlab-unit-primitive": "unknown"},
            None,
            None,
            "400: This endpoint cannot be used for unknown purpose",
        ),
    ],
)
@pytest.mark.usefixtures("patch_unit_primitive", "patch_feature_category")
async def test_feature_categories(
    headers,
    expected_error,
    expected_unit_primitive,
    expected_category,
):
    @feature_categories(
        {
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_1: DummyGitLabFeatureCategory.AWESOME_CATEGORY_1,
            DummyGitLabUnitPrimitive.AWESOME_FEATURE_2: DummyGitLabFeatureCategory.AWESOME_CATEGORY_2,
        }
    )
    async def to_be_decorated(_request: Request):
        pass

    request = Mock(spec=Request)
    request.headers = headers

    if expected_error:
        with pytest.raises(HTTPException, match=expected_error):
            await to_be_decorated(request)
    else:
        with patch("ai_gateway.api.feature_category.context") as mock_context:
            await to_be_decorated(request)

            mock_context.__setitem__.assert_has_calls(
                [
                    mock.call(
                        "meta.feature_category",
                        expected_category,
                    ),
                    mock.call(
                        "meta.unit_primitive",
                        expected_unit_primitive,
                    ),
                ]
            )


def _passthrough_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize(
    ("decorators", "expected"),
    [
        ([], frozenset()),
        (
            [feature_category(GitLabFeatureCategory.DUO_CHAT)],
            frozenset({GitLabFeatureCategory.DUO_CHAT}),
        ),
        (
            [
                feature_categories(
                    {
                        GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
                        GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
                    }
                )
            ],
            frozenset(
                {
                    GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
                    GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
                }
            ),
        ),
        # Categories stay visible under other decorators, as on the proxy endpoints
        (
            [
                _passthrough_decorator,
                feature_category(GitLabFeatureCategory.DUO_CHAT),
            ],
            frozenset({GitLabFeatureCategory.DUO_CHAT}),
        ),
    ],
)
def test_get_feature_categories(decorators, expected):
    async def endpoint():
        pass

    # Apply bottom-up, matching how stacked decorators are applied
    for decorator in reversed(decorators):
        endpoint = decorator(endpoint)

    assert get_feature_categories(endpoint) == expected


@pytest.mark.parametrize(
    ("unit_primitive", "expected_category"),
    [
        (GitLabUnitPrimitive.COMPLETE_CODE, "code_suggestions"),
        (GitLabUnitPrimitive.EXPLAIN_VULNERABILITY, "vulnerability_management"),
        # Not a GitLabFeatureCategory member, so it's recorded as the catalog's string
        (GitLabUnitPrimitive.DUO_AGENT_PLATFORM, "duo_workflow"),
    ],
)
def test_set_feature_category_for_unit_primitive(unit_primitive, expected_category):
    with request_cycle_context({"meta.feature_category": "ai_abstraction_layer"}):
        set_feature_category_for_unit_primitive(unit_primitive)

        assert current_feature_category() == expected_category


def test_set_feature_category_for_unit_primitive_not_in_catalog():
    with (
        patch(
            "ai_gateway.api.feature_category.get_catalog",
            return_value=Mock(unit_primitives=[]),
        ),
        request_cycle_context({"meta.feature_category": "ai_abstraction_layer"}),
    ):
        set_feature_category_for_unit_primitive(GitLabUnitPrimitive.COMPLETE_CODE)

        assert current_feature_category() == "ai_abstraction_layer"


def test_catalog_is_loaded_once():
    with (
        patch("ai_gateway.api.feature_category.load_catalog") as mock_load_catalog,
        request_cycle_context({}),
    ):
        set_feature_category_for_unit_primitive(GitLabUnitPrimitive.COMPLETE_CODE)
        set_feature_category_for_unit_primitive(GitLabUnitPrimitive.DUO_CHAT)

    mock_load_catalog.assert_not_called()
    assert get_catalog() is get_catalog()
