from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH


class NoLanggraphLangchainImportsChecker(BaseChecker):
    _FORBIDDEN_IMPORTS = ["langgraph", "langchain"]
    _WORKFLOWS_PATH = "duo_workflow_service/workflows"

    name = "no-langgraph-langchain-imports"
    priority = HIGH
    msgs = {
        "W9001": (
            "Import from langgraph or langchain is not allowed in duo_workflow_service/workflows",
            "no-langgraph-langchain-imports",
            "No imports from langgraph or langchain should be used in duo_workflow_service/workflows",
        ),
    }

    def visit_importfrom(self, node):
        if (
            self.linter.current_file
            and "duo_workflow_service/workflows" in self.linter.current_file
            and self._import_forbidden(node.modname)
        ):
            self.add_message("no-langgraph-langchain-imports", node=node)

    def visit_import(self, node):
        if (
            self.linter.current_file
            and "duo_workflow_service/workflows" in self.linter.current_file
        ):
            for name, _ in node.names:
                if self._import_forbidden(name):
                    self.add_message("no-langgraph-langchain-imports", node=node)
                    break

    def _import_forbidden(self, import_name) -> bool:
        for forbidden_import in self._FORBIDDEN_IMPORTS:
            if forbidden_import in import_name:
                return True
        return False


def register(linter):
    linter.register_checker(NoLanggraphLangchainImportsChecker(linter))
