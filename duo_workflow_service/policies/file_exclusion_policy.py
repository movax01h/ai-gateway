from pathlib import Path
from typing import List

import gitmatch

from duo_workflow_service.gitlab.gitlab_api import Project
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

CONTEXT_EXCLUSION_MESSAGE = "excluded due to policy"


class FileExclusionPolicy:
    def __init__(self, project: Project):
        self.project = project
        exclusion_rules = project and (project.get("exclusion_rules") or [])
        # Ensure exclusion_rules is a list of strings
        self._exclusion_rules: List[str] = (
            exclusion_rules if isinstance(exclusion_rules, list) else []
        )
        self._matcher = (
            gitmatch.compile(self._exclusion_rules) if self._exclusion_rules else None
        )

    def is_allowed(self, filename: str) -> bool:
        """Check if a single file matches any exclusion pattern."""
        if not is_feature_enabled(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION):
            return True

        if not self._matcher:
            return True

        # Normalize path separators to forward slashes
        filename = str(Path(filename)).replace("\\", "/").lstrip("/")

        # Use gitmatch to check if the file should be ignored
        return not self._matcher.match(filename)

    def filter_allowed(self, filenames: List[str]):
        """Filter a list of filenames, returning only those allowed by the policy."""
        if not is_feature_enabled(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION):
            return filenames, []

        if not self._matcher:
            return filenames, []

        allowed_files = []
        excluded_files = []

        for filename in filenames:
            filename = filename.strip()
            if filename and self.is_allowed(filename):
                allowed_files.append(filename)
            elif filename:
                excluded_files.append(filename)

        return allowed_files, excluded_files

    @staticmethod
    def format_user_exclusion_message(blocked_files: List[str]) -> str:
        if blocked_files:
            return f" - files excluded:\n{"\n".join(blocked_files)}"

        return ""

    @staticmethod
    def format_llm_exclusion_message(blocked_files: List[str]) -> str:
        if blocked_files:
            return f"Files excluded due to policy, continue without files:\n{"\n".join(blocked_files)}"

        return ""

    @staticmethod
    def is_allowed_for_project(project, filename: str):
        policy = FileExclusionPolicy(project)
        return policy.is_allowed(filename)
