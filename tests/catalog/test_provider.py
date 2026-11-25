import base64
import json
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import SecretStr

from ai_gateway.catalog import (
    CatalogDocuments,
    CatalogLoadError,
    FileCatalogProvider,
    RepoCatalogProvider,
    SingleFileCatalogProvider,
    catalog_provider_from_config,
)
from ai_gateway.config import ConfigModelCatalog

MODELS = {"models": [{"gitlab_identifier": "model-1"}]}
UNIT_PRIMITIVES = {
    "configurable_unit_primitives": [{"feature_setting": "chat"}],
    "flows": {"chat": ["chat"]},
}
RESTRICTIONS = {"restricted_models": [{"identifier": "model-1"}]}
CATALOG = {**MODELS, **UNIT_PRIMITIVES, **RESTRICTIONS}
TOKEN = "glpat-secret-token"


@pytest.fixture(name="catalog_files")
def catalog_files_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "models_path": tmp_path / "models.yml",
        "unit_primitives_path": tmp_path / "unit_primitives.yml",
        "model_restrictions_path": tmp_path / "model_restrictions.yml",
    }
    for path, document in zip(paths.values(), (MODELS, UNIT_PRIMITIVES, RESTRICTIONS)):
        path.write_text(yaml.safe_dump(document))
    return paths


class TestCatalogDocuments:
    def test_from_catalog_document_splits_into_the_three_documents(self):
        documents = CatalogDocuments.from_catalog_document(CATALOG, "v1")

        assert documents == CatalogDocuments(
            models=MODELS,
            unit_primitives=UNIT_PRIMITIVES,
            model_restrictions=RESTRICTIONS,
            version="v1",
        )

    def test_from_catalog_document_defaults_optional_sections(self):
        documents = CatalogDocuments.from_catalog_document(
            {
                "models": [],
                "configurable_unit_primitives": [],
                "restricted_models": None,
            },
            "v1",
        )

        assert documents.unit_primitives == {"configurable_unit_primitives": []}
        assert documents.model_restrictions == {"restricted_models": []}

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ([], "must be a mapping"),
            (None, "must be a mapping"),
            ({"configurable_unit_primitives": []}, r"missing keys: \['models'\]"),
            (
                {**CATALOG, "default_models": {}},
                r"unknown keys: \['default_models'\]",
            ),
        ],
    )
    def test_from_catalog_document_rejects_bad_shapes(self, document, message):
        with pytest.raises(CatalogLoadError, match=message):
            CatalogDocuments.from_catalog_document(document, "v1")


class TestFileCatalogProvider:
    def test_load_reads_the_three_files(self, catalog_files):
        documents = FileCatalogProvider(**catalog_files).load()

        assert documents.models == MODELS
        assert documents.unit_primitives == UNIT_PRIMITIVES
        assert documents.model_restrictions == RESTRICTIONS
        assert documents.version

    def test_version_follows_the_content(self, catalog_files):
        provider = FileCatalogProvider(**catalog_files)
        first = provider.load().version

        assert provider.load().version == first

        catalog_files["model_restrictions_path"].write_text("restricted_models: []\n")

        assert provider.load().version != first

    def test_empty_restrictions_file_gives_an_empty_document(self, catalog_files):
        catalog_files["model_restrictions_path"].write_text("")

        assert FileCatalogProvider(**catalog_files).load().model_restrictions == {}

    def test_invalid_yaml_names_the_file(self, catalog_files):
        catalog_files["models_path"].write_text("models: [")

        with pytest.raises(CatalogLoadError, match="models.yml is not valid YAML"):
            FileCatalogProvider(**catalog_files).load()

    def test_missing_file_raises(self, catalog_files):
        catalog_files["models_path"].unlink()

        with pytest.raises(FileNotFoundError):
            FileCatalogProvider(**catalog_files).load()

    def test_storage(self, catalog_files):
        assert FileCatalogProvider(**catalog_files).storage == "file"


def test_single_file_catalog_provider_loads_a_catalog_file(tmp_path: Path):
    path = tmp_path / "catalog.yml"
    path.write_text(yaml.safe_dump(CATALOG))

    documents = SingleFileCatalogProvider(path).load()

    assert documents.models == MODELS
    assert documents.unit_primitives == UNIT_PRIMITIVES
    assert documents.model_restrictions == RESTRICTIONS
    assert documents.version
    assert SingleFileCatalogProvider(path).storage == "file"


RELEASES_PATH = b"/api/v4/projects/group%2Fcatalog/releases"
FILE_PATH = b"/api/v4/projects/group%2Fcatalog/repository/files/config%2Fcatalog.yml"


class TestRepoCatalogProvider:
    @staticmethod
    def _provider(handler, ref: str | None = None) -> RepoCatalogProvider:
        return RepoCatalogProvider(
            gitlab_url="https://gitlab.example.com/",
            project="group/catalog",
            ref=ref,
            path="config/catalog.yml",
            token=SecretStr(TOKEN),
            transport=httpx.MockTransport(handler),
        )

    @staticmethod
    def _file_response(content: str, commit: str = "abc123") -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "file_path": "config/catalog.yml",
                "content": base64.b64encode(content.encode()).decode(),
                "encoding": "base64",
                "last_commit_id": commit,
            },
        )

    @classmethod
    def _repo(
        cls,
        releases: httpx.Response | None = None,
        file: httpx.Response | None = None,
    ):
        """A handler that answers the releases and file endpoints and records each request."""
        requests: list[httpx.Request] = []
        releases = releases or httpx.Response(200, json=[{"tag_name": "v42"}])
        file = file or cls._file_response(yaml.safe_dump(CATALOG))

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/releases"):
                return releases
            return file

        return handler, requests

    def test_load_reads_the_file_at_the_latest_release(self):
        handler, requests = self._repo()

        documents = self._provider(handler).load()

        assert documents == CatalogDocuments(
            models=MODELS,
            unit_primitives=UNIT_PRIMITIVES,
            model_restrictions=RESTRICTIONS,
            version="v42",
        )
        assert [request.url.raw_path for request in requests] == [
            RELEASES_PATH + b"?order_by=released_at&sort=desc&per_page=1",
            FILE_PATH + b"?ref=v42",
        ]
        assert all(request.method == "GET" for request in requests)
        assert all(request.headers["PRIVATE-TOKEN"] == TOKEN for request in requests)

    def test_load_reads_the_file_at_a_pinned_ref(self):
        handler, requests = self._repo()

        documents = self._provider(handler, ref="v41").load()

        assert documents.version == "abc123"
        assert [request.url.raw_path for request in requests] == [
            FILE_PATH + b"?ref=v41"
        ]

    def test_an_empty_ref_reads_the_latest_release(self):
        handler, _ = self._repo()

        assert self._provider(handler, ref="").load().version == "v42"

    @pytest.mark.parametrize(
        ("releases", "message"),
        [
            (httpx.Response(200, json=[]), "has no releases"),
            (httpx.Response(404), "HTTP 404 for the latest release"),
            (httpx.Response(200, text="not json"), "unexpected response"),
            (httpx.Response(200, json={"tag_name": "v42"}), "unexpected response"),
            (httpx.Response(200, json=[{"name": "v42"}]), "unexpected response"),
            (httpx.Response(200, json=[{"tag_name": ""}]), "unexpected response"),
            (httpx.Response(200, json=[{"tag_name": 42}]), "unexpected response"),
        ],
    )
    def test_load_rejects_bad_release_responses(self, releases, message):
        handler, requests = self._repo(releases=releases)

        with pytest.raises(CatalogLoadError, match=message) as exc_info:
            self._provider(handler).load()

        assert TOKEN not in str(exc_info.value)
        assert len(requests) == 1

    @pytest.mark.parametrize(
        ("file", "message"),
        [
            (httpx.Response(404), "HTTP 404 for config/catalog.yml at v42"),
            (httpx.Response(401), "HTTP 401"),
            (httpx.Response(200, text="not json"), "unexpected response"),
            (httpx.Response(200, json={"content": "x"}), "unexpected response"),
            (
                httpx.Response(
                    200, json={"content": "not base64!", "last_commit_id": "abc"}
                ),
                "unexpected response",
            ),
        ],
    )
    def test_load_rejects_bad_file_responses(self, file, message):
        handler, _ = self._repo(file=file)

        with pytest.raises(CatalogLoadError, match=message) as exc_info:
            self._provider(handler).load()

        assert TOKEN not in str(exc_info.value)

    def test_load_with_a_pinned_ref_needs_the_last_commit(self):
        handler, _ = self._repo(
            file=httpx.Response(
                200, json={"content": base64.b64encode(b"models: []").decode()}
            )
        )

        with pytest.raises(CatalogLoadError, match="unexpected response"):
            self._provider(handler, ref="v41").load()

    def test_load_wraps_transport_errors(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        with pytest.raises(
            CatalogLoadError, match="Could not reach the catalog repository"
        ) as exc_info:
            self._provider(handler).load()

        assert TOKEN not in str(exc_info.value)

    def test_load_rejects_invalid_yaml(self):
        handler, _ = self._repo(file=self._file_response("models: ["))

        with pytest.raises(CatalogLoadError, match="not valid YAML"):
            self._provider(handler).load()

    def test_load_rejects_a_catalog_with_the_wrong_shape(self):
        handler, _ = self._repo(file=self._file_response(json.dumps({"models": []})))

        with pytest.raises(CatalogLoadError, match="missing keys"):
            self._provider(handler).load()

    def test_storage(self):
        assert self._provider(lambda _request: httpx.Response(200)).storage == "repo"


class TestCatalogProviderFromConfig:
    def test_file_storage_uses_the_shipped_files(self, catalog_files):
        shipped = FileCatalogProvider(**catalog_files)

        assert catalog_provider_from_config(ConfigModelCatalog(), shipped) is shipped

    def test_repo_storage_reads_the_repository(self, catalog_files):
        provider = catalog_provider_from_config(
            ConfigModelCatalog(
                storage="repo",
                repo_project="group/catalog",
                repo_token=SecretStr(TOKEN),
            ),
            FileCatalogProvider(**catalog_files),
        )

        assert isinstance(provider, RepoCatalogProvider)

    def test_repo_storage_without_a_token_is_rejected(self, catalog_files):
        # ConfigModelCatalog rejects this on validation; model_construct skips it.
        config = ConfigModelCatalog.model_construct(
            storage="repo", repo_project="group/catalog", repo_token=None
        )

        with pytest.raises(ValueError, match="needs repo_token"):
            catalog_provider_from_config(config, FileCatalogProvider(**catalog_files))
