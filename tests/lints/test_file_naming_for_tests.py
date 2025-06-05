import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import astroid
import pylint.testutils
import pytest

from lints import file_naming_for_tests


@pytest.fixture
def node():
    node = astroid.extract_node("def test():\n  pass")
    node.file = f"{os.getcwd()}/tests/path/to/test_filename.py"
    return node


class TestFileNamingForTests(pylint.testutils.CheckerTestCase):

    CHECKER_CLASS = file_naming_for_tests.FileNamingForTests

    @patch("lints.file_naming_for_tests.Path")
    def test_valid_test_file(self, mock_path_class, node):
        """Test that a valid test file doesn't trigger the warning."""
        mock_exist_path_instance = MagicMock(spec=Path)
        mock_exist_path_instance.is_file.return_value = True

        mock_non_exist_path_instance = MagicMock(spec=Path)
        mock_non_exist_path_instance.is_file.return_value = False

        mock_path_class.side_effect = [
            mock_non_exist_path_instance,
            mock_exist_path_instance,
        ]

        with self.assertNoMessages():
            self.checker.visit_module(node)

        mock_path_class.assert_has_calls(
            [call("ai_gateway/path/to/filename.py"), call("./path/to/filename.py")],
            any_order=True,
        )

    @patch("pathlib.Path.is_file")
    def test_invalid_test_file(self, mock_is_file, node):
        """Test that an invalid test file triggers the warning."""

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="W5003",
                node=node,
            ),
            ignore_position=True,
        ):
            mock_is_file.return_value = False
            self.checker.visit_module(node)

    @patch("pathlib.Path.is_file")
    def test_excluded_file(self, mock_is_file, node):
        """Test that excluded files don't trigger the warning."""

        node.file = f"{os.getcwd()}/tests/test_structured_log.py"

        with self.assertNoMessages():
            mock_is_file.return_value = False
            self.checker.visit_module(node)

    @patch("pathlib.Path.is_file")
    def test_non_test_file(self, mock_is_file, node):
        """Test that non-test files don't trigger the warning."""

        node.file = f"{os.getcwd()}/ai_gateway/foo.py"

        with self.assertNoMessages():
            mock_is_file.return_value = False
            self.checker.visit_module(node)
