import os
from pathlib import Path

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter

# DO NOT ADD FILES FROM THE ai_gateway MODULE
EXCLUDED_FILES = {
    "/tests/code_suggestions/test_authentication.py",
    "/tests/code_suggestions/test_engine.py",
    "/tests/code_suggestions/test_instrumentators.py",
    "/tests/code_suggestions/test_logging.py",
    "/tests/code_suggestions/test_processing.py",
    "/tests/prompts/test_litellm_prompt.py",
    "/tests/searches/test_search_container.py",
    "/tests/test_structured_log.py",
}

# Folders to scan for implementation files
# All AI Gateway service related files is nested under `ai_gateway` folder.
# Others are directly under root folder, for example, `lints`, `eval` & `duo-workflow-service`.
SOURCE_DIRS = {
    ".",
    "ai_gateway",
}


class FileNamingForTests(BaseChecker):
    name = "file-naming-for-tests"
    msgs = {
        "W5003": (
            "Test file name does not match the file it is testing.",
            "file-naming-for-tests",
            "Test files must be name to the file they are testing: tests/path/to/test_filename.py must "
            "test path/to/filename.py. See https://docs.gitlab.com/development/python_guide/styleguide/",
        )
    }

    def visit_module(self, node: nodes.Module) -> None:
        file_path = node.file.replace(os.getcwd(), "")

        # Optimize order of checks: cheapest first
        if (
            file_path in EXCLUDED_FILES
            or not file_path.startswith("/tests/")
            or "test_" not in file_path
        ):
            return

        relative_test_path = file_path[len("/tests/") :].replace("test_", "", 1)

        if not any(
            Path(f"{source_dir}/{relative_test_path}").is_file()
            for source_dir in SOURCE_DIRS
        ):
            self.add_message(
                "W5003",
                node=node,
            )


def register(linter: "PyLinter") -> None:
    linter.register_checker(FileNamingForTests(linter))
