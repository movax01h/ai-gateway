import astroid
import pylint.testutils
from astroid import nodes

from lints import no_exception_swallowing_in_tools


class TestNoExceptionSwallowingInTools(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = no_exception_swallowing_in_tools.NoExceptionSwallowingInToolsChecker

    def test_valid_execute_that_propagates_exceptions(self):
        """Test that _execute() methods that let exceptions propagate are valid."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            result = await some_operation()
            return result
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        with self.assertNoMessages():
            self.checker.visit_asyncfunctiondef(node)

    def test_valid_execute_with_proper_exception_raising(self):
        """Test that _execute() methods that raise ToolException are valid."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception as e:
                raise ToolException(f"Failed: {e}")
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        with self.assertNoMessages():
            self.checker.visit_asyncfunctiondef(node)

    def test_valid_execute_outside_tools_directory(self):
        """Test that _execute() methods outside duo_workflow_service/tools/ are ignored."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return json.dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "ai_gateway/something/my_file.py"

        with self.assertNoMessages():
            self.checker.visit_asyncfunctiondef(node)

    def test_valid_non_execute_method_with_try_except(self):
        """Test that non-_execute() methods can catch exceptions and return JSON."""
        node = astroid.extract_node("""
        async def some_helper(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return json.dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        with self.assertNoMessages():
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_basic_exception_swallowing_with_json_dumps(self):
        """Test that basic try/except with json.dumps({"error": ...}) is invalid."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return json.dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract nodes
        try_node = list(node.nodes_of_class(nodes.Try))[0]
        handler_node = try_node.handlers[0]
        return_nodes = list(handler_node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_exception_swallowing_with_dumps_imported(self):
        """Test that using dumps() directly (imported) is also caught."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract nodes
        try_node = list(node.nodes_of_class(nodes.Try))[0]
        handler_node = try_node.handlers[0]
        return_nodes = list(handler_node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_conditional_return_in_handler(self):
        """Test that conditional returns inside exception handlers are caught."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception as e:
                if isinstance(e, ValueError):
                    return json.dumps({"error": "validation error"})
                raise
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the return node - handler has raise so only W9003 fires
        return_nodes = list(node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[1]  # Second return has error JSON

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_nested_try_blocks(self):
        """Test that nested try blocks with exception swallowing are caught."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                try:
                    result = await some_operation()
                    return result
                except ValueError:
                    return json.dumps({"error": "nested error"})
            except Exception:
                raise
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the INNER exception handler node (the one with ValueError)
        try_nodes = list(node.nodes_of_class(nodes.Try))
        inner_try_node = try_nodes[1]  # The nested try block
        handler_node = inner_try_node.handlers[0]
        return_nodes = list(handler_node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_return_inside_loop_in_handler(self):
        """Test that returns inside loops in exception handlers are caught."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                for attempt in range(3):
                    if attempt == 2:
                        return json.dumps({"error": "all attempts failed"})
                raise
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the return node - handler has raise so only W9003 fires
        return_nodes = list(node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[1]  # Second return has error JSON

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_exception_without_raise_even_without_error_key(self):
        """Test that except handlers without raise are invalid regardless of return value."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return json.dumps({"status": "failed", "message": "something"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the exception handler node
        try_node = list(node.nodes_of_class(nodes.Try))[0]
        handler_node = try_node.handlers[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_sync_function_def(self):
        """Test that sync _execute() methods are also checked."""
        node = astroid.extract_node("""
        def _execute(self, input_data: str) -> str:  #@
            try:
                result = some_operation()
                return result
            except Exception:
                return json.dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract nodes
        try_node = list(node.nodes_of_class(nodes.Try))[0]
        handler_node = try_node.handlers[0]
        return_nodes = list(handler_node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_functiondef(node)

    def test_invalid_error_json_return_outside_except(self):
        """Test that returning error JSON outside except block triggers W9003."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            if not input_data:
                return json.dumps({"error": "validation failed"})
            return await some_operation()
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the return node with error JSON
        return_nodes = list(node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]  # First return has error JSON

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_error_json_with_dict_constructor(self):
        """Test that dict(error=...) constructor is also caught."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            if not input_data:
                return json.dumps(dict(error="validation failed"))
            return await some_operation()
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract the return node with error JSON
        return_nodes = list(node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_invalid_double_violation_except_and_error_json(self):
        """Test that except without raise triggers both W9002 and W9003."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            try:
                result = await some_operation()
                return result
            except Exception:
                return json.dumps({"error": "something went wrong"})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        # Extract nodes
        try_node = list(node.nodes_of_class(nodes.Try))[0]
        handler_node = try_node.handlers[0]
        return_nodes = list(handler_node.nodes_of_class(nodes.Return))
        error_return_node = return_nodes[0]

        with self.assertAddsMessages(
            pylint.testutils.MessageTest(
                msg_id="exception-swallowing-in-tool",
                node=handler_node,
            ),
            pylint.testutils.MessageTest(
                msg_id="error-json-return-in-tool",
                node=error_return_node,
            ),
            ignore_position=True,
        ):
            self.checker.visit_asyncfunctiondef(node)

    def test_valid_success_json_return(self):
        """Test that success JSON returns (no error key) outside except are valid."""
        node = astroid.extract_node("""
        async def _execute(self, input_data: str) -> str:  #@
            result = await some_operation()
            return json.dumps({"status": "success", "data": result})
        """)
        self.checker.linter.current_file = "duo_workflow_service/tools/my_tool.py"

        with self.assertNoMessages():
            self.checker.visit_asyncfunctiondef(node)
