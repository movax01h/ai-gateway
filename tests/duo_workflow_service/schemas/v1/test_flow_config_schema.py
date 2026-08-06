import json

from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    INPUT_JSONSCHEMA_VERSION,
    FlowConfig,
)
from duo_workflow_service.schemas.v1 import flow_config_schema
from duo_workflow_service.schemas.v1.flow_config_schema import (
    generate_flow_config_schema,
    write_flow_config_schema,
)


class TestGenerateFlowConfigSchema:
    def test_includes_schema_version(self):
        schema = generate_flow_config_schema()

        assert schema["$schema"] == INPUT_JSONSCHEMA_VERSION

    def test_matches_flow_config_model_json_schema(self):
        """Everything but the injected ``$schema`` key must mirror the live Pydantic model.

        This is the invariant the CI lint job (``make check-duo-workflow-flow-schema``)
        relies on: if ``FlowConfig`` changes without regenerating the checked-in file,
        the two fall out of sync and the job fails.
        """
        schema = generate_flow_config_schema()
        schema_without_version = {k: v for k, v in schema.items() if k != "$schema"}

        assert schema_without_version == FlowConfig.model_json_schema()

    def test_describes_flow_config_shape(self):
        schema = generate_flow_config_schema()

        assert schema["title"] == "FlowConfig"
        assert schema["type"] == "object"
        assert set(schema["required"]) == {
            "flow",
            "components",
            "routers",
            "environment",
            "version",
        }

    def test_is_json_serializable(self):
        schema = generate_flow_config_schema()

        # Raises if the schema contains anything json.dumps can't handle (e.g. a
        # non-string dict key or a non-serializable value slipping in from Pydantic).
        json.dumps(schema)

    def test_schema_version_overrides_any_key_from_flow_config_model_json_schema(
        self, monkeypatch
    ):
        """INPUT_JSONSCHEMA_VERSION must win even if Pydantic starts emitting its own ``$schema`` key.

        Regression test for the intended dict-merge order: ``FlowConfig.model_json_schema()``
        is spread first and ``$schema`` is set after, so it always takes precedence.
        """
        monkeypatch.setattr(
            FlowConfig,
            "model_json_schema",
            classmethod(lambda cls: {"$schema": "bogus"}),
        )

        schema = generate_flow_config_schema()

        assert schema["$schema"] == INPUT_JSONSCHEMA_VERSION

    def test_checked_in_schema_is_up_to_date(self):
        """Guard against a stale checked-in schema file, independent of the CI job.

        The CI lint job relies on a ``changes:`` file list to decide when to run, which
        can miss upstream drift (e.g. a Pydantic version bump changing
        ``model_json_schema()`` output) that doesn't touch any listed file. This test
        runs on every ``make test`` regardless, so it always catches a stale file.
        """
        assert (
            json.loads(flow_config_schema.SCHEMA_PATH.read_text(encoding="utf-8"))
            == generate_flow_config_schema()
        ), "Schema is stale. Run `make duo-workflow-flow-schema` and commit the result."


class TestWriteFlowConfigSchema:
    def test_writes_generated_schema_to_schema_path(self, tmp_path, monkeypatch):
        destination = tmp_path / "flow_config.schema.json"
        monkeypatch.setattr(flow_config_schema, "SCHEMA_PATH", destination)

        write_flow_config_schema()

        assert json.loads(destination.read_text(encoding="utf-8")) == (
            generate_flow_config_schema()
        )

    def test_writes_trailing_newline(self, tmp_path, monkeypatch):
        destination = tmp_path / "flow_config.schema.json"
        monkeypatch.setattr(flow_config_schema, "SCHEMA_PATH", destination)

        write_flow_config_schema()

        assert destination.read_text(encoding="utf-8").endswith("\n")

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        destination = tmp_path / "flow_config.schema.json"
        destination.write_text("stale content", encoding="utf-8")
        monkeypatch.setattr(flow_config_schema, "SCHEMA_PATH", destination)

        write_flow_config_schema()

        assert json.loads(destination.read_text(encoding="utf-8")) == (
            generate_flow_config_schema()
        )
