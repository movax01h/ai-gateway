# pylint: disable=file-naming-for-tests

import os
from unittest.mock import patch

import astroid
import pylint.testutils
import pytest

from lints import file_naming_for_tests


@pytest.fixture
def node():
    node = astroid.extract_node("def test():\n  pass")
    node.file = f"{os.getcwd()}/tests/path/to/test_filename.py"
    return node


class TestFileNameingForTests(pylint.testutils.CheckerTestCase):

    CHECKER_CLASS = file_naming_for_tests.FileNamingForTests

    @patch("os.path.exists")
    def test_valid_test_file(self, file_exists, node):
        """Test that a valid test file doesn't trigger the warning."""
        # Create a module node with a valid test file path

        # Create the corresponding file that is being tested
        with self.assertNoMessages():
            file_exists.return_value = True
            self.checker.visit_module(node)

            file_exists.assert_called_with("ai_gateway/path/to/filename.py")

    @patch("os.path.exists")
    def test_invalid_test_file(self, file_exists, node):
        """Test that an invalid test file triggers the warning."""

        # The file being tested doesn't exist
        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="W5003",
                node=node,
            ),
            ignore_position=True,
        ):
            file_exists.return_value = False
            self.checker.visit_module(node)

    @patch("os.path.exists")
    def test_excluded_file(self, file_exists, node):
        """Test that excluded files don't trigger the warning."""
        node.file = f"{os.getcwd()}/tests/test_structured_log.py"

        # The file is in the excluded list
        with self.assertNoMessages():
            file_exists.return_value = False
            self.checker.visit_module(node)

    @patch("os.path.exists")
    def test_non_test_file(self, file_exists, node):
        """Test that non-test files don't trigger the warning."""
        node.file = f"{os.getcwd()}/ai_gateway/foo.py"

        # The file is in the excluded list
        with self.assertNoMessages():
            file_exists.return_value = False
            self.checker.visit_module(node)
