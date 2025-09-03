from typing import get_type_hints

from contract.contract_pb2 import OsInformationContext as OsInformationContextProto
from contract.contract_pb2 import (
    ShellInformationContext as ShellInformationContextProto,
)
from duo_workflow_service.workflows.type_definitions import (
    OsInformationContext,
    ShellInformationContext,
)


class TestContextContract:
    """Test that Pydantic model fields match protobuf generated class fields."""

    def test_os_information_context(self):
        pydantic_fields = OsInformationContext.model_fields.keys()

        proto_fields = list(
            field.name for field in OsInformationContextProto.DESCRIPTOR.fields
        )

        # Verify that they have the same field names
        assert len(pydantic_fields) == len(proto_fields)
        assert set(pydantic_fields) == set(proto_fields)

        # Verify that the fields have the same type
        test_data = {"platform": "Linux", "architecture": "x86_64"}

        # Create instances
        pydantic_instance = OsInformationContext(**test_data)
        proto_instance = OsInformationContextProto(**test_data)

        # Verify each field can hold the same type of data
        for field_name, field_info in OsInformationContext.model_fields.items():
            # Check field exists in proto
            assert hasattr(proto_instance, field_name)

            # Get values from both instances
            pydantic_value = getattr(pydantic_instance, field_name)
            proto_value = getattr(proto_instance, field_name)

            # Check types match
            assert type(pydantic_value) is type(proto_value)

    def test_shell_information_context(self):
        pydantic_fields = ShellInformationContext.model_fields.keys()

        proto_fields = list(
            field.name for field in ShellInformationContextProto.DESCRIPTOR.fields
        )

        # Verify that they have the same field names
        assert len(pydantic_fields) == len(proto_fields)
        assert set(pydantic_fields) == set(proto_fields)

        # Verify that the fields have the same type
        test_data = {
            "shell_name": "bash",
            "shell_type": "unix",
            "shell_variant": "5.1.8",
            "shell_environment": "native",
            "ssh_session": False,
            "cwd": "/home/user/project",
        }

        # Create instances
        pydantic_instance = ShellInformationContext(**test_data)
        proto_instance = ShellInformationContextProto(**test_data)

        # Verify each field can hold the same type of data
        for field_name, field_info in ShellInformationContext.model_fields.items():
            # Check field exists in proto
            assert hasattr(proto_instance, field_name)

            # Get values from both instances
            pydantic_value = getattr(pydantic_instance, field_name)
            proto_value = getattr(proto_instance, field_name)

            # Check types match
            assert type(pydantic_value) is type(proto_value)
