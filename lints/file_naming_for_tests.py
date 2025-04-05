import os

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter

# DO NOT ADD FILES
EXCLUDED_FILES = {
    "/tests/test_structured_log.py",
    "/tests/searches/test_search_container.py",
    "/tests/code_suggestions/test_instrumentators.py",
    "/tests/code_suggestions/test_engine.py",
    "/tests/code_suggestions/test_processing.py",
    "/tests/code_suggestions/test_logging.py",
    "/tests/code_suggestions/test_prompts.py",
    "/tests/code_suggestions/test_authentication.py",
    "/tests/code_suggestions/test_generation.py",
    "/tests/code_suggestions/models/test_anthropic.py",
    "/tests/code_suggestions/models/test_palm.py",
    "/tests/code_suggestions/models/test_mock.py",
    "/tests/code_suggestions/models/test_base.py",
    "/tests/prompts/test_litellm_prompt.py",
}


class FileNamingForTests(BaseChecker):
    name = "file-naming-for-tests"
    msgs = {
        "W5003": (
            "Test file name does not match the file it is testing.",
            "file-naming-for-tests",
            "Test files must be name to the file they are testing: tests/path/to/test_filename.py must "
            "test tests/path/to/filename.py. See https://docs.gitlab.com/development/python_guide/styleguide/",
        )
    }

    def visit_module(self, node: nodes.Module) -> None:
        file_path = node.file.replace(os.getcwd(), "")

        if (
            not file_path.startswith("/tests/")
            or "test_" not in file_path
            or file_path in EXCLUDED_FILES
        ):
            return

        expected_path = (
            f"ai_gateway{file_path.replace('tests/', '').replace('test_', '')}"
        )

        if not os.path.exists(expected_path):
            self.add_message(
                "W5003",
                node=node,
            )


def register(linter: "PyLinter") -> None:
    linter.register_checker(FileNamingForTests(linter))
