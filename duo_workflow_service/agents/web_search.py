"""Web-search availability, resolved once per agent build."""

from dataclasses import dataclass

from duo_workflow_service.client_capabilities import is_client_capable
from lib.feature_flags.context import FeatureFlag, is_feature_enabled


@dataclass(frozen=True)
class WebSearchState:
    """Defaults to "web search plays no part here", for callers that never resolved it."""

    supported: bool = False
    """Web search exists for this session at all, regardless of the user's toggle."""

    active: bool = False
    """Supported AND the user has web search toggled on right now."""

    @classmethod
    def resolve(
        cls, user_toggle: bool, group_allowed: bool = False
    ) -> "WebSearchState":
        """Resolve web-search availability for the current session.

        Args:
            user_toggle: Whether the user has web search toggled on for this request.
            group_allowed: Whether the top-level group allows web search at all.

        Returns:
            A state where `supported` reflects the group setting, the feature flag and
            the client capability, and `active` additionally requires `user_toggle`.
        """
        supported = (
            group_allowed
            and is_feature_enabled(FeatureFlag.DAP_WEB_SEARCH)
            and is_client_capable("web_search")
        )
        return cls(supported=supported, active=supported and user_toggle)
