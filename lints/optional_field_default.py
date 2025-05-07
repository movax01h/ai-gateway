import re

from pylint.checkers import BaseChecker

OPTIONAL_TYPE_PATTERN = re.compile(r"^Optional\[(.+)\]$")


class OptionalFieldDefault(BaseChecker):
    name = "optional-field-default"

    msgs = {
        "C9001": (
            "Optional field '%s' should have a default value of None or use `default=None` in Field(...)",
            "optional-field-missing-default",
            "Optional fields should explicitly default to None to be truly optional in Pydantic.",
        ),
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self._warned_locations = set()

    def visit_annassign(self, node):
        file_path = node.root().file

        location = (file_path, node.lineno, node.col_offset)

        if location in self._warned_locations:
            return

        if not self._should_check(node):
            return

        has_none_default = self._has_none_default(node.value)

        if not has_none_default:
            self._warned_locations.add(location)
            self._warn(node)

    def _should_check(self, node):
        if not (
            hasattr(node, "annotation")
            and hasattr(node, "value")
            and node.value is not None
        ):
            return False

        if not (hasattr(node.value, "func") and hasattr(node.value, "args")):
            return False

        func = node.value.func
        if not getattr(func, "name", None) == "Field":
            return False

        try:
            annotation = node.annotation.as_string()
            return bool(OPTIONAL_TYPE_PATTERN.match(annotation))
        except Exception:
            return False

    def _has_none_default(self, value_node):
        for kw in getattr(value_node, "keywords", []):
            if kw.arg == "default" and getattr(kw.value, "value", None) is None:
                return True

        args = value_node.args
        if args and getattr(args[0], "value", None) is None:
            return True

        return False

    def _warn(self, node):
        field_name = getattr(getattr(node, "target", None), "name", "<unknown>")
        self.add_message(
            "optional-field-missing-default", node=node, args=(field_name,)
        )


def register(linter):
    linter.register_checker(OptionalFieldDefault(linter))
