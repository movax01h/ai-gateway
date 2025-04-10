from typing import Any, Optional

from langchain.tools import BaseTool

from duo_workflow_service.gitlab.http_client import GitlabHttpClient


class DuoBaseTool(BaseTool):
    @property
    def gitlab_client(self) -> GitlabHttpClient:
        client = self.metadata.get("gitlab_client")  # type: ignore
        if not client:
            raise RuntimeError("gitlab_client is not set")
        return client

    @property
    def gitlab_host(self) -> str:
        host = self.metadata.get("gitlab_host")  # type: ignore
        if not host:
            raise RuntimeError("gitlab_host is not set")
        return host

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool can only be run asynchronously")

    def format_display_message(self, args: Any) -> Optional[str]:
        # Handle both dictionary and Pydantic model arguments
        if isinstance(args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        elif hasattr(args, "dict"):
            # Handle Pydantic model instances
            args_str = ", ".join(f"{k}={v}" for k, v in args.dict().items())
        else:
            args_str = str(args)

        return f"Using {self.name}: {args_str}"
