from typing import Any

__all__: list[str] = []

# Moved to the code_review feature package. Resolve lazily, so importing this package at startup does not warn.
_MOVED_TOOLS = {
    "BuildReviewMergeRequestContext": "ai.features.review.code_review.components",
    "DetectReviewContextTruncation": "ai.features.review.code_review.components",
    "FinalizeCodeReviewFindings": "ai.features.review.code_review.components",
    "PostDuoCodeReview": "ai.features.review.code_review.components",
    "PostDuoCodeReviewFindings": "ai.features.review.code_review.components",
}


def __getattr__(name: str) -> Any:
    module_path = _MOVED_TOOLS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib  # pylint: disable=import-outside-toplevel
    import warnings  # pylint: disable=import-outside-toplevel

    warnings.warn(
        f"duo_workflow_service.tools.code_review.{name} moved to {module_path}",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    # Keep the moved names visible to dir() and enumeration-based tooling.
    return sorted(set(globals()) | set(_MOVED_TOOLS))
