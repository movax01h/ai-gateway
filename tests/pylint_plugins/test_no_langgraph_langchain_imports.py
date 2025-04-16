import astroid
import astroid.nodes
import pylint.testutils
import pytest

from duo_workflow_service.pylint_plugins.no_langgraph_langchain_imports import (
    NoLanggraphLangchainImportsChecker,
)


class TestUniqueReturnChecker(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = NoLanggraphLangchainImportsChecker

    @pytest.mark.parametrize(
        "imports, file_path",
        [
            ("from langgraph import Graph", "duo_workflow_service/workflows/test.py"),
            ("from langchain import OpenAI", "duo_workflow_service/workflows/test.py"),
        ],
    )
    def test_importfrom_violation(self, imports, file_path):
        import_node = astroid.extract_node(imports)
        import_node.root().name = file_path
        self.checker.visit_importfrom(import_node)
        if isinstance(import_node, astroid.nodes.NodeNG):
            self.assertAddsMessages(
                pylint.testutils.MessageTest(
                    msg_id="no-langgraph-langchain-imports",
                    node=import_node,
                )
            )

    @pytest.mark.parametrize(
        "imports, file_path",
        [
            ("from langgraph import Graph", "duo_workflow_service/other_file.py"),
            (
                "from langchain_core.messages import AIMessage",
                "duo_workflow_service/other_file.py",
            ),
            ("from langchain_core.messages import AIMessage", "other_package/test.py"),
            ("from langgraph import Graph", "other_package/test.py"),
        ],
    )
    def test_no_importfrom_violation(self, imports, file_path):
        import_node = astroid.extract_node(imports)
        import_node.root().name = file_path
        with self.assertNoMessages():
            self.checker.visit_importfrom(import_node)

    @pytest.mark.parametrize(
        "imports, file_path",
        [
            ("import langgraph", "duo_workflow_service/workflows/test.py"),
            ("import langchain", "duo_workflow_service/workflows/test.py"),
            ("import langchain_core", "duo_workflow_service/workflows/test.py"),
        ],
    )
    def test_import_violation(self, imports, file_path):
        import_node = astroid.extract_node(imports)
        import_node.root().name = file_path
        self.checker.visit_import(import_node)
        if isinstance(import_node, astroid.nodes.NodeNG):
            self.assertAddsMessages(
                pylint.testutils.MessageTest(
                    msg_id="no-langgraph-langchain-imports",
                    node=import_node,
                )
            )

    @pytest.mark.parametrize(
        "imports, file_path",
        [
            ("import langgraph", "duo_workflow_service/other_file.py"),
            ("import langchain_core.messages", "duo_workflow_service/other_file.py"),
            ("import langchain_core.messages", "other_package/test.py"),
            ("import langgraph", "other_package/test.py"),
        ],
    )
    def test_no_import_violation(self, imports, file_path):
        import_node = astroid.extract_node(imports)
        import_node.root().name = file_path
        with self.assertNoMessages():
            self.checker.visit_import(import_node)
