"""Validation rules for lists of history optimizer configs."""

from collections import Counter

from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    HistoryOptimizerConfig,
    LegacyTrimConfig,
)


def validate_history_optimizer_configs(
    configs: list[HistoryOptimizerConfig],
) -> None:
    """Validate a list of history optimizer configs.

    Enforces:
    - At most one of {``CompactionConfig``, ``LegacyTrimConfig``}.
    - No duplicate optimizer types.

    Empty lists are permitted; callers apply a default (typically
    ``[LegacyTrimConfig()]``). No ordering constraints between optimizer
    types are enforced — order comes from the caller's list.

    Args:
        configs: List of optimizer configs to validate.

    Raises:
        ValueError: If validation fails.
    """
    types = [type(c) for c in configs]

    # Check duplicates first so that two configs of the same type (e.g.
    # [CompactionConfig(), CompactionConfig()]) surface the accurate
    # "duplicate" error rather than the misleading mutual-exclusion error
    # that would otherwise fire because two CompactionConfigs both count
    # toward the "main" cap.
    counts = Counter(types)
    duplicates = [t.__name__ for t, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate history_optimizers types not allowed: {duplicates}"
        )

    main_types = {CompactionConfig, LegacyTrimConfig}
    main_count = sum(1 for t in types if t in main_types)
    if main_count > 1:
        raise ValueError(
            "history_optimizers may include at most one of: "
            "CompactionConfig, LegacyTrimConfig"
        )
