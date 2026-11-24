"""Tests for the GitLab docs ingestion data store step."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from google.api_core.exceptions import AlreadyExists, FailedPrecondition

STEP_PATH = (
    Path(__file__).parents[5]
    / "scripts"
    / "ingest"
    / "gitlab-docs"
    / "steps"
    / "create_data_store.py"
)

IMPORT_ERROR = FailedPrecondition("Documents cannot be imported for this project.")


@pytest.fixture(name="step")
def step_fixture(monkeypatch):
    monkeypatch.delenv("INGEST_DRY_RUN", raising=False)
    monkeypatch.setenv("GCP_PROJECT_NAME", "test-project")
    monkeypatch.setenv("DATA_STORE_ID", "gitlab-docs-19-7")
    monkeypatch.setenv("BIGQUERY_DATASET_ID", "gitlab_docs_19_7")
    monkeypatch.setenv("BIGQUERY_TABLE_NAME", "2026-11-24-03-14-09")

    spec = importlib.util.spec_from_file_location(
        "gitlab_docs_create_data_store", STEP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(name="steps")
def steps_fixture(step):
    with (
        patch.object(step, "create_data_store") as create_data_store,
        patch.object(step, "import_documents") as import_documents,
        patch.object(step, "delete_data_store") as delete_data_store,
    ):
        yield create_data_store, import_documents, delete_data_store


class TestRun:
    def test_keeps_new_data_store_when_import_succeeds(self, step, steps):
        _, import_documents, delete_data_store = steps

        step.run()

        import_documents.assert_called_once_with()
        delete_data_store.assert_not_called()

    def test_deletes_new_data_store_when_import_fails(self, step, steps):
        _, import_documents, delete_data_store = steps
        import_documents.side_effect = IMPORT_ERROR

        with pytest.raises(FailedPrecondition):
            step.run()

        delete_data_store.assert_called_once_with()

    def test_keeps_existing_data_store_when_import_fails(self, step, steps):
        create_data_store, import_documents, delete_data_store = steps
        create_data_store.side_effect = AlreadyExists("already exists")
        import_documents.side_effect = IMPORT_ERROR

        with pytest.raises(FailedPrecondition):
            step.run()

        delete_data_store.assert_not_called()


class TestDeleteDataStore:
    def test_deletes_the_data_store_of_the_run(self, step):
        with patch.object(
            step.discoveryengine, "DataStoreServiceClient"
        ) as client_class:
            step.delete_data_store()

        client = client_class.return_value
        request = client.delete_data_store.call_args.kwargs["request"]

        assert request.name == (
            "projects/test-project/locations/global/collections/default_collection/"
            "dataStores/gitlab-docs-19-7"
        )
        client.delete_data_store.return_value.result.assert_called_once_with()
