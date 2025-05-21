import asyncio
import os

from duo_workflow_service.gitlab.direct_http_client import DirectGitLabHttpClient
from duo_workflow_service.gitlab.executor_http_client import ExecutorGitLabHttpClient
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


def get_http_client(
    outbox: asyncio.Queue, inbox: asyncio.Queue, base_url: str, gitlab_token: str
) -> GitlabHttpClient:
    if base_url == os.getenv("DUO_WORKFLOW_DIRECT_CONNECTION_BASE_URL"):
        return DirectGitLabHttpClient(base_url, gitlab_token)
    else:
        return ExecutorGitLabHttpClient(outbox, inbox)
