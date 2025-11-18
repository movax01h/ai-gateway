from typing import List, Tuple

from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy


class DiffExclusionPolicy(FileExclusionPolicy):
    def filter_allowed_diffs(self, diffs: List[dict]) -> Tuple[List[dict], List[str]]:
        """Filter a list of diff dictionaries, removing files that match exclusion patterns.

        Args:
            diffs: List of dictionaries, each containing diff information

        Returns:
            Tuple of (filtered_diffs, excluded_files)
        """
        if not self._matcher:
            return diffs, []

        filtered_diffs = []
        excluded_files = []

        for diff_data in diffs:
            # Extract old_path and new_path
            old_path = diff_data.get("old_path", "")
            new_path = diff_data.get("new_path", "")

            # Check if both paths are allowed
            if self.is_allowed(old_path) and self.is_allowed(new_path):
                filtered_diffs.append(diff_data)
            else:
                # Add excluded files to the list (avoid duplicates)
                if (
                    old_path
                    and not self.is_allowed(old_path)
                    and old_path not in excluded_files
                ):
                    excluded_files.append(old_path)
                if (
                    new_path
                    and not self.is_allowed(new_path)
                    and new_path not in excluded_files
                ):
                    excluded_files.append(new_path)

        return filtered_diffs, excluded_files
