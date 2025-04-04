from unittest.mock import MagicMock

import pytest

from ai_gateway.api.server_utils import extract_retry_after_header
from ai_gateway.models.base import ModelAPICallError


@pytest.fixture
def headers():
    return {"retry-after": "30"}


@pytest.fixture
def exception(headers):
    mock_exc = MagicMock(spec=ModelAPICallError)

    mock_error = MagicMock()

    mock_response = MagicMock()
    mock_response.headers = headers
    mock_error.response = mock_response

    mock_exc.errors = [mock_error]

    return mock_exc


@pytest.mark.parametrize(
    ("headers", "expected"), [({"retry-after": "30"}, "30"), ({}, None)]
)
def test_extract_retry_after_header_with_valid_header(expected, exception):
    result = extract_retry_after_header(exception)

    assert result == expected


def test_extract_retry_after_header_with_no_header(exception):
    exception.errors[0].response.headers = {}

    result = extract_retry_after_header(exception)

    assert result is None


def test_extract_retry_after_header_with_no_response(exception):
    delattr(exception.errors[0], "response")

    result = extract_retry_after_header(exception)

    assert result is None


def test_extract_retry_after_header_with_empty_errors(exception):
    exception.errors = []

    result = extract_retry_after_header(exception)

    assert result is None


def test_extract_retry_after_header_with_no_errors_attribute(exception):
    delattr(exception, "errors")

    result = extract_retry_after_header(exception)
    assert result is None
