import asyncio
from unittest.mock import patch

import pytest

from duo_workflow_service.gitlab.direct_http_client import DirectGitLabHttpClient
from duo_workflow_service.gitlab.executor_http_client import ExecutorGitLabHttpClient
from duo_workflow_service.gitlab.http_client_factory import get_http_client


@pytest.fixture
def queues():
    return asyncio.Queue(), asyncio.Queue()


def test_get_http_client_custom_gitlab(queues):
    """Test that get_http_client returns ExecutorGitLabHttpClient for custom GitLab instances"""
    outbox, inbox = queues
    base_url = "https://custom.gitlab.example.com"
    gitlab_token = "test-token"

    client = get_http_client(outbox, inbox, base_url, gitlab_token)

    assert isinstance(client, ExecutorGitLabHttpClient)
    assert client.outbox == outbox
    assert client.inbox == inbox


def test_get_http_client_with_env_var(queues):
    """Test that the factory respects the DUO_WORKFLOW_DIRECT_CONNECTION_BASE_URL environment variable"""
    outbox, inbox = queues
    custom_base_url = "https://custom.direct.gitlab"
    gitlab_token = "test-token"

    with patch.dict(
        "os.environ", {"DUO_WORKFLOW_DIRECT_CONNECTION_BASE_URL": custom_base_url}
    ):
        # Should return DirectGitLabHttpClient when base_url matches env var
        client = get_http_client(outbox, inbox, custom_base_url, gitlab_token)
        assert isinstance(client, DirectGitLabHttpClient)
        assert client.base_url == custom_base_url
        assert client.gitlab_token == gitlab_token

        # Should return ExecutorGitLabHttpClient for other URLs
        other_url = "https://other.gitlab"
        client = get_http_client(outbox, inbox, other_url, gitlab_token)
        assert isinstance(client, ExecutorGitLabHttpClient)
        assert client.outbox == outbox
        assert client.inbox == inbox
