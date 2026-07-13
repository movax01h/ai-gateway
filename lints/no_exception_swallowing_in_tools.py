"""Pylint checker to prevent tools from swallowing exceptions.

This checker ensures that tool _execute() methods do not catch exceptions
without re-raising them, which prevents proper error handling by
ToolNodeWithErrorCorrection.

See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1974
"""

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH
from pylint.lint import PyLinter


class NoExceptionSwallowingInToolsChecker(BaseChecker):
    """Checker to prevent tools from catching exceptions without re-raising."""

    name = "no-exception-swallowing-in-tools"
    priority = HIGH

    msgs = {
        "W9002": (
            "Tool _execute() method catches exceptions without re-raising. "
            "Let exceptions propagate instead.",
            "exception-swallowing-in-tool",
            "Tools should not catch exceptions and swallow them. "
            "Instead, let exceptions propagate naturally (or catch and re-raise) "
            "so that upstream nodes can handle them properly. "
            "See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1974",
        ),
        "W9003": (
            "Tool _execute() method returns JSON error payload. "
            "Raise ToolException instead.",
            "error-json-return-in-tool",
            "Tools should not return json.dumps({'error': ...}). "
            "Instead, raise ToolException so that upstream nodes can handle errors properly. "
            "See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1974",
        ),
    }

    def visit_asyncfunctiondef(self, node: nodes.AsyncFunctionDef) -> None:
        """Check async _execute() methods in tools for exception swallowing."""
        self._check_execute_method(node)

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        """Check _execute() methods in tools for exception swallowing."""
        self._check_execute_method(node)

    def _check_execute_method(self, node) -> None:
        """Check if this is an _execute method that swallows exceptions."""
        # Only check _execute methods in duo_workflow_service/tools/
        if node.name != "_execute":
            return

        current_file = self.linter.current_file
        if not current_file or "duo_workflow_service/tools/" not in current_file:
            return

        # Check for try/except blocks without raise
        self._check_for_exception_swallowing(node)

        # Check for error JSON returns anywhere in the method
        self._check_for_error_json_returns(node)

    def _check_for_exception_swallowing(self, node: nodes.NodeNG) -> None:
        """Recursively check node for exception swallowing pattern."""
        # Check if this is a try/except node
        if isinstance(node, nodes.Try):
            for handler in node.handlers:
                # Flag if handler doesn't re-raise the exception
                if not self._has_raise_statement(handler):
                    self.add_message("exception-swallowing-in-tool", node=handler)

            # Recurse into try body, orelse, finally to find nested try blocks
            for child in node.body:
                self._check_for_exception_swallowing(child)
            for child in node.orelse:
                self._check_for_exception_swallowing(child)
            for child in node.finalbody:
                self._check_for_exception_swallowing(child)
        else:
            # For non-Try nodes, recurse into all children
            for child in node.get_children():
                self._check_for_exception_swallowing(child)

    @staticmethod
    def _has_raise_statement(handler: nodes.ExceptHandler) -> bool:
        """Check if exception handler has a raise statement."""
        for _ in handler.nodes_of_class(nodes.Raise):
            return True
        return False

    def _check_for_error_json_returns(self, node: nodes.NodeNG) -> None:
        """Check for return json.dumps({"error": ...}) anywhere in the method."""
        # Find all return statements in the function
        for return_node in node.nodes_of_class(nodes.Return):
            if self._is_json_error_return(return_node):
                self.add_message("error-json-return-in-tool", node=return_node)

    def _is_json_error_return(self, node: nodes.Return) -> bool:  # noqa: PLR0911  # AST shape guards
        """Check if return statement is: return json.dumps({"error": ...})."""
        # pylint: disable=too-many-return-statements,too-many-branches
        if node.value is None:
            return False

        # Check if it's a Call node (function call)
        if not isinstance(node.value, nodes.Call):
            return False

        # Check if it's calling json.dumps() or just dumps()
        func = node.value.func
        if isinstance(func, nodes.Attribute) and func.attrname == "dumps":
            # json.dumps(...)
            pass
        elif isinstance(func, nodes.Name) and func.name == "dumps":
            # dumps(...) - likely imported
            pass
        else:
            return False

        # Check if first argument is a dict with "error" key
        if not node.value.args:
            return False

        first_arg = node.value.args[0]

        # Check literal dict: {"error": ...}
        if isinstance(first_arg, nodes.Dict):
            # first_arg.items is a list of (key, value) tuples
            for key, _ in first_arg.items:
                if isinstance(key, nodes.Const) and key.value == "error":
                    return True

        # Check dict() constructor: dict(error=...)
        if isinstance(first_arg, nodes.Call):
            # Check if calling dict()
            if isinstance(first_arg.func, nodes.Name) and first_arg.func.name == "dict":
                # Check keyword args for error=...
                if first_arg.keywords:
                    for keyword in first_arg.keywords:
                        if keyword.arg == "error":
                            return True

        return False


def register(linter: PyLinter) -> None:
    """Register the checker with pylint."""
    linter.register_checker(NoExceptionSwallowingInToolsChecker(linter))
