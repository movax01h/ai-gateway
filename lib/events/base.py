from typing import Self

from duo_workflow_service.agent_platform.utils import parse_workflow_definition


class GLReportingEventContext:
    """Stores flow context for legacy and Flow Registry definitions used in UsageQuota and Billing events.

    This class replaces the deprecated CategoryEnum and provides a unified way to handle:
    - Legacy workflow types (e.g., "software_development", "chat")
    - Flow Registry definitions with versioning (e.g., "my_flow/v1")
    - AI Catalog items (flows with flow_config)

    The class maintains backward compatibility with CategoryEnum through the `value` property, which returns the
    legacy workflow type string that can be used in places expecting CategoryEnum values.

    Examples:
        Legacy workflow:
            >>> context = GLReportingEventContext.from_workflow_definition("software_development")
            >>> context.value
            'software_development'
            >>> context.feature_qualified_name
            'software_development'
            >>> context.feature_ai_catalog_item
            False

        Flow Registry versioned:
            >>> context = GLReportingEventContext.from_workflow_definition("my_flow/v1")
            >>> context.value
            'my_flow'
            >>> context.feature_qualified_name
            'my_flow/v1'
            >>> context.feature_ai_catalog_item
            False
    """

    def __init__(
        self, legacy_workflow_type: str, flow_definition: str, is_ai_catalog_item: bool
    ):
        """Initialize a GLReportingEventContext instance.

        Args:
            legacy_workflow_type: The legacy workflow type string (e.g., "software_development").
            flow_definition: The full flow definition string (e.g., "my_flow/v1").
            is_ai_catalog_item: Whether this flow is an AI Catalog item.
        """
        self._legacy_workflow_type = legacy_workflow_type
        self._flow_definition = flow_definition
        self._is_ai_catalog_item = is_ai_catalog_item

    @property
    def value(self) -> str:
        """Get the legacy workflow type for backward compatibility.

        Returns:
            The legacy workflow type string (e.g., "software_development", "my_flow").
        """
        return self._legacy_workflow_type

    @property
    def feature_qualified_name(self) -> str:
        """Get the fully qualified flow definition name.

        Returns:
            The full flow definition string including version if applicable (e.g., "my_flow/v1").
        """
        return self._flow_definition

    @property
    def feature_ai_catalog_item(self) -> bool:
        """Check if this flow is an AI Catalog item.

        AI Catalog items are flows that have an associated flow_config stored in Rails,
        indicating they're part of the AI Catalog system.

        Returns:
            True if this is an AI Catalog item, False otherwise.
        """
        return self._is_ai_catalog_item

    def __str__(self):
        return self._legacy_workflow_type

    def __eq__(self, other):
        if isinstance(other, GLReportingEventContext):
            return self._legacy_workflow_type == other._legacy_workflow_type
        return self._legacy_workflow_type == other

    @classmethod
    def from_workflow_definition(
        cls, value: str | None, has_flow_config: bool = False
    ) -> Self:
        """Create a GLReportingEventContext from a workflow definition string.

        This is the primary factory method for creating GLReportingEventContext instances. Consider calling
        resolve_workflow_definition() first to provide backward compatibility for empty workflow definitions
        (defaults to "software_development").

        Args:
            value: The workflow definition string. Can be a legacy type ("software_development", "chat") or a Flow
                Registry path ("my_flow/v1"). If None, defaults to "software_development" for backward compatibility.
            has_flow_config: Whether this flow has an associated flow configuration, marking it as an AI Catalog item.

        Returns:
            A GLReportingEventContext instance configured based on the input.
        """
        if not value:
            # backward compatibility for old GitLab instances
            value = "software_development"

        try:
            _, legacy_workflow_type = parse_workflow_definition(value)
            new_flow_type = value
        except ValueError:
            legacy_workflow_type = value
            new_flow_type = value

        return cls(legacy_workflow_type, new_flow_type, has_flow_config)
