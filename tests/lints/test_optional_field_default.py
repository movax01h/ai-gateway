import astroid
import pytest
from pylint.testutils import CheckerTestCase, MessageTest

from lints.optional_field_default import OptionalFieldDefault


class TestOptionalFieldDefault(CheckerTestCase):
    CHECKER_CLASS = OptionalFieldDefault

    @pytest.mark.parametrize(
        "code_snippet, expected_field_name",
        [
            (
                "from typing import Optional\n"
                "from pydantic import Field\n"
                "class MyModel:\n"
                "    my_field: Optional[int] = Field(...)",
                "my_field",
            ),
            (
                "from typing import Optional\n"
                "from pydantic import Field\n"
                "class MyModel:\n"
                "    foo: Optional[str] = Field(description='something')",
                "foo",
            ),
        ],
    )
    def test_optional_field_missing_default(self, code_snippet, expected_field_name):
        node = astroid.extract_node(code_snippet)
        field_node = node.body[0]
        self.checker.visit_annassign(field_node)
        self.assertAddsMessages(
            MessageTest(
                msg_id="optional-field-missing-default",
                node=field_node,
                args=(expected_field_name,),
            )
        )

    @pytest.mark.parametrize(
        "code_snippet",
        [
            (
                "from typing import Optional\n"
                "from pydantic import Field\n"
                "class MyModel:\n"
                "    my_field: Optional[int] = Field(default=None)"
            ),
            (
                "from typing import Optional\n"
                "from pydantic import Field\n"
                "class MyModel:\n"
                "    my_field: Optional[int] = Field(None)"
            ),
        ],
    )
    def test_optional_field_with_default_none(self, code_snippet):
        node = astroid.extract_node(code_snippet)
        with self.assertNoMessages():
            self.checker.visit_annassign(node)

    def test_non_optional_field_not_checked(self):
        code = (
            "from pydantic import Field\n"
            "class MyModel:\n"
            "    foo: int = Field(...)"
        )
        node = astroid.extract_node(code)
        with self.assertNoMessages():
            self.checker.visit_annassign(node)
