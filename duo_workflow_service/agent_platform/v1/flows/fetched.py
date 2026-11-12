"""A saved AI Catalog flow that the service fetches from GitLab by workflow ID."""

import hmac
from typing import Any, Callable, Optional, override

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogAgentsUnavailableError,
    CatalogItemsError,
    CatalogItemsFetchError,
)
from duo_workflow_service.agent_platform.v1.catalog.fetch import fetch_catalog_flow
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.errors.typing import (
    NotifiableAgentException,
    WorkflowTokenRejectedError,
)
from duo_workflow_service.flow_config_digest import definition_digest
from duo_workflow_service.security.exceptions import SecurityException

__all__ = [
    "FetchedFlow",
    "FetchedFlowConfigError",
    "FetchedFlowDigestMismatchError",
    "FetchedFlowFetchError",
    "FlowConfigBuilder",
]

FlowConfigBuilder = Callable[[dict[str, Any]], FlowConfig]


class FetchedFlowConfigError(NotifiableAgentException):
    """The flow definition GitLab served cannot be built into a flow."""

    UI_MESSAGE = (
        "The saved flow definition that GitLab returned for this session is not valid. "
        "Check the flow in the AI Catalog."
    )

    def __init__(self, internal_detail: str) -> None:
        super().__init__(self.UI_MESSAGE, internal_detail=internal_detail)


class FetchedFlowFetchError(NotifiableAgentException):
    """GitLab could not serve the saved flow, for a reason other than 404."""

    UI_MESSAGE = (
        "This session stopped because GitLab could not load its saved flow. Try again."
    )

    def __init__(self, internal_detail: str) -> None:
        super().__init__(self.UI_MESSAGE, internal_detail=internal_detail)


class FetchedFlowDigestMismatchError(NotifiableAgentException):
    """The flow definition the run received is not the one its token was issued for."""

    UI_MESSAGE = (
        "The flow definition that this session received does not match the flow it was "
        "started for, so the session stopped."
    )

    def __init__(self, internal_detail: str) -> None:
        super().__init__(self.UI_MESSAGE, internal_detail=internal_detail)


class FetchedFlow(Flow):
    """A v1 flow whose config comes from GitLab instead of the start request.

    The service cannot call GitLab until the run's stream is open, so the flow is created without a config. Before
    the graph is compiled, :meth:`_prepare_compile` fetches the catalog items of the run once. The root definition
    becomes the config, and the agents in the same response bind to its ``ai-catalog`` entries.
    """

    def __init__(
        self,
        *,
        config_builder: FlowConfigBuilder,
        authorized_flow_config_digest: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create the flow without a config.

        Args:
            config_builder: Builds and checks a config from the fetched definition. It raises ``ValueError``,
                ``TypeError`` or ``SecurityException`` when the definition cannot be used.
            authorized_flow_config_digest: The ``flow_config_digest`` claim of the run's token. The fetched definition
                must match it. ``None`` only for a debug user, whose run is not checked.
            **kwargs: The arguments of :class:`Flow`, without ``config``.
        """
        super().__init__(config=None, **kwargs)
        self._config_builder = config_builder
        self._authorized_flow_config_digest = authorized_flow_config_digest

    @override
    def _registry_flow_id(self) -> Optional[str]:
        # A saved catalog flow is never a registry flow, whatever its definition says, so it gets no registry grant.
        return None

    @override
    async def _prepare_compile(self) -> None:
        """Fetch the flow definition and its AI Catalog agents with one request, then use them.

        Raises:
            CatalogFlowUnavailableError: If GitLab does not serve a definition for the run.
            FetchedFlowFetchError: If GitLab answers with another error status, or a body that is not an object.
            CatalogAgentsUnavailableError: If GitLab refused an AI Catalog agent of the flow, or an agent is not valid.
            FetchedFlowDigestMismatchError: If the definition does not match the token's digest.
            FetchedFlowConfigError: If the definition cannot be built into a flow.
        """
        try:
            fetched = await fetch_catalog_flow(self._http_client, self._workflow_id)
        except CatalogItemsFetchError as exc:
            raise FetchedFlowFetchError(str(exc)) from exc
        except CatalogItemsError as exc:
            raise CatalogAgentsUnavailableError.from_error(exc) from exc
        self._check_digest(fetched.definition)

        try:
            config = self._config_builder(fetched.definition)
            # _use_config registers the definition's inline response schemas, which raise
            # ValueError for a bad schema, so it belongs in the same guard as the builder.
            self._use_config(config)
        except (ValueError, TypeError, SecurityException) as exc:
            raise FetchedFlowConfigError(
                f"Fetched flow definition for workflow {self._workflow_id} is not valid: {exc}"
            ) from exc

        if self._references_ai_catalog():
            self._catalog_items = self._catalog_items.with_ai_catalog_agents(
                fetched.agents.agents, fetched.agents.skipped_reason
            )

    def _check_digest(self, definition: dict[str, Any]) -> None:
        """Compare the fetched definition with the digest that Rails put in the token.

        The fetch is a RunHTTPRequest. Workhorse serves it when the client connects through Workhorse, but a client that
        connects to the service directly answers it itself. The digest makes sure that such a client cannot change the
        flow.
        """
        expected = self._authorized_flow_config_digest
        if expected is None:
            return

        actual = definition_digest(definition)
        if hmac.compare_digest(expected.encode(), actual.encode()):
            return

        self.log.warning(
            "Fetched flow definition does not match the token digest",
            workflow_id=self._workflow_id,
            expected_digest=expected,
            actual_digest=actual,
        )
        # Raise with a WorkflowTokenRejectedError cause so the server maps last_error to
        # PERMISSION_DENIED with the rejection's fixed gRPC detail. Without a cause the
        # NotifiableAgentException leaves last_error None, and the run ends as UNKNOWN with
        # no terminal status. This is the last_error path, not the preauth token_rejection
        # path, so it does not count the reject metric or record a WORKFLOW_REJECT event.
        # The digest values stay in the internal detail.
        raise FetchedFlowDigestMismatchError(
            f"Fetched flow definition for workflow {self._workflow_id} has digest {actual}, "
            f"but the token allows {expected}"
        ) from WorkflowTokenRejectedError(FetchedFlowDigestMismatchError.UI_MESSAGE)
