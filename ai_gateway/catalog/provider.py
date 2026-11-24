"""Where the model catalog is read from.

The model catalog is the data in ``models.yml``, ``unit_primitives.yml`` and ``model_restrictions.yml``. A
``CatalogProvider`` reads it and hands back the parsed YAML documents; ``ModelSelectionConfig`` validates them and
builds its in-memory snapshot. By default it reads the files shipped in the image (``file``). GitLab.com can read
the catalog from a repository through the GitLab API instead (``repo``).
"""

import base64
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
import yaml
from pydantic import SecretStr

from ai_gateway.config import ConfigModelCatalog

__all__ = [
    "CATALOG_DOCUMENT_KEYS",
    "CatalogDocuments",
    "CatalogLoadError",
    "CatalogProvider",
    "FileCatalogProvider",
    "RepoCatalogProvider",
    "SingleFileCatalogProvider",
    "catalog_provider_from_config",
]

# The top-level keys of a single-file catalog and the shipped file each one comes from.
CATALOG_DOCUMENT_KEYS: dict[str, str] = {
    "models": "models.yml",
    "configurable_unit_primitives": "unit_primitives.yml",
    "flows": "unit_primitives.yml",
    "restricted_models": "model_restrictions.yml",
}


class CatalogLoadError(Exception):
    """The catalog could not be read or does not have the expected shape.

    The message never includes credentials or catalog contents, so it is safe to log.
    """


@dataclass(frozen=True)
class CatalogDocuments:
    """The parsed YAML of one catalog version, before any validation.

    Attributes:
        models: The ``models.yml`` document (``{"models": [...]}``).
        unit_primitives: The ``unit_primitives.yml`` document (``configurable_unit_primitives`` and ``flows``).
        model_restrictions: The ``model_restrictions.yml`` document (``{"restricted_models": [...]}``).
        version: Identifies this version of the catalog: for ``repo``, the release tag, or the SHA of the last commit
            that changed the file when a ref is pinned; a content hash otherwise.
    """

    models: dict[str, Any]
    unit_primitives: dict[str, Any]
    model_restrictions: dict[str, Any] = field(default_factory=dict)
    version: str = ""

    @classmethod
    def from_catalog_document(cls, document: Any, version: str) -> "CatalogDocuments":
        """Split a single-file catalog into the three documents.

        A single-file catalog has the top-level keys of the three shipped files in one document (see
        ``CATALOG_DOCUMENT_KEYS``); ``models`` and ``configurable_unit_primitives`` are required.

        Raises:
            CatalogLoadError: If the document is not a mapping, misses a required key or has an unknown one.
        """
        if not isinstance(document, dict):
            raise CatalogLoadError("Catalog document must be a mapping")

        unknown = sorted(set(document) - CATALOG_DOCUMENT_KEYS.keys())
        if unknown:
            raise CatalogLoadError(f"Catalog document has unknown keys: {unknown}")

        missing = sorted({"models", "configurable_unit_primitives"} - document.keys())
        if missing:
            raise CatalogLoadError(f"Catalog document is missing keys: {missing}")

        unit_primitives = {
            "configurable_unit_primitives": document["configurable_unit_primitives"]
        }
        if "flows" in document:
            unit_primitives["flows"] = document["flows"]

        return cls(
            models={"models": document["models"]},
            unit_primitives=unit_primitives,
            model_restrictions={
                "restricted_models": document.get("restricted_models") or []
            },
            version=version,
        )


def _content_version(*contents: bytes) -> str:
    digest = hashlib.sha256()
    for content in contents:
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()[:12]


def _parse_yaml(content: bytes, source: str) -> Any:
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise CatalogLoadError(f"{source} is not valid YAML") from exc


class CatalogProvider(ABC):
    """Reads the model catalog from its storage."""

    @property
    @abstractmethod
    def storage(self) -> str:
        """The name of the storage, for logs and metrics (``file`` or ``repo``)."""

    @abstractmethod
    def load(self) -> CatalogDocuments:
        """Read the current catalog.

        Raises:
            CatalogLoadError: If the catalog cannot be read or parsed.
        """


class FileCatalogProvider(CatalogProvider):
    """Reads the catalog from ``models.yml``, ``unit_primitives.yml`` and ``model_restrictions.yml`` on disk."""

    def __init__(
        self,
        models_path: Path,
        unit_primitives_path: Path,
        model_restrictions_path: Path,
    ) -> None:
        self._models_path = models_path
        self._unit_primitives_path = unit_primitives_path
        self._model_restrictions_path = model_restrictions_path

    @property
    def storage(self) -> str:
        return "file"

    def load(self) -> CatalogDocuments:
        # Missing or unreadable shipped files are a broken image, not a catalog error, so OSError propagates.
        paths = (
            self._models_path,
            self._unit_primitives_path,
            self._model_restrictions_path,
        )
        contents = [path.read_bytes() for path in paths]
        models, unit_primitives, model_restrictions = (
            _parse_yaml(content, path.name) for content, path in zip(contents, paths)
        )
        return CatalogDocuments(
            models=models,
            unit_primitives=unit_primitives,
            model_restrictions=model_restrictions or {},
            version=_content_version(*contents),
        )


class SingleFileCatalogProvider(CatalogProvider):
    """Reads a single-file catalog (see ``CatalogDocuments.from_catalog_document``) from disk.

    Used to validate a catalog file, for example in the catalog repository's CI.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def storage(self) -> str:
        return "file"

    def load(self) -> CatalogDocuments:
        content = self._path.read_bytes()
        return CatalogDocuments.from_catalog_document(
            _parse_yaml(content, self._path.name), _content_version(content)
        )


class RepoCatalogProvider(CatalogProvider):
    """Reads a single-file catalog from a GitLab repository through the GitLab API.

    By default it reads the file at the tag of the repository's latest release (by ``released_at``), and the
    version is that tag. With ``ref``, it reads the file at that tag, branch or commit instead, and the version is
    the SHA of the last commit that changed the file there.

    Reads are synchronous: they run at startup and, from a later phase, on a background refresh, never on the
    request path.
    """

    def __init__(
        self,
        *,
        gitlab_url: str,
        project: str,
        path: str,
        token: SecretStr,
        ref: Optional[str] = None,
        timeout: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        project_url = (
            f"{gitlab_url.rstrip('/')}/api/v4/projects/{quote(project, safe='')}"
        )
        self._releases_url = f"{project_url}/releases"
        self._file_url = f"{project_url}/repository/files/{quote(path, safe='')}"
        self._ref = ref or None
        self._path = path
        self._token = token
        self._timeout = timeout
        self._transport = transport

    @property
    def storage(self) -> str:
        return "repo"

    def load(self) -> CatalogDocuments:
        try:
            with httpx.Client(
                timeout=self._timeout,
                transport=self._transport,
                headers={"PRIVATE-TOKEN": self._token.get_secret_value()},
            ) as client:
                ref = self._ref or self._latest_release_tag(client)
                body = self._get_json(
                    client, self._file_url, {"ref": ref}, f"{self._path} at {ref}"
                )
        except httpx.HTTPError as exc:
            raise CatalogLoadError(
                f"Could not reach the catalog repository: {type(exc).__name__}"
            ) from exc

        try:
            content = base64.b64decode(body["content"], validate=True)
            version = ref if self._ref is None else str(body["last_commit_id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise CatalogLoadError(
                "Catalog repository returned an unexpected response"
            ) from exc

        return CatalogDocuments.from_catalog_document(
            _parse_yaml(content, self._path), version
        )

    def _latest_release_tag(self, client: httpx.Client) -> str:
        releases = self._get_json(
            client,
            self._releases_url,
            {"order_by": "released_at", "sort": "desc", "per_page": 1},
            "the latest release",
        )
        if releases == []:
            raise CatalogLoadError("Catalog repository has no releases")
        try:
            tag = releases[0]["tag_name"]
        except (KeyError, TypeError, IndexError) as exc:
            raise CatalogLoadError(
                "Catalog repository returned an unexpected response"
            ) from exc
        if not isinstance(tag, str) or not tag:
            raise CatalogLoadError("Catalog repository returned an unexpected response")
        return tag

    @staticmethod
    def _get_json(
        client: httpx.Client, url: str, params: dict[str, Any], what: str
    ) -> Any:
        response = client.get(url, params=params)
        if response.status_code != httpx.codes.OK:
            raise CatalogLoadError(
                f"Catalog repository returned HTTP {response.status_code} for {what}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise CatalogLoadError(
                "Catalog repository returned an unexpected response"
            ) from exc


def catalog_provider_from_config(
    config: ConfigModelCatalog, shipped_files: FileCatalogProvider
) -> CatalogProvider:
    """Return the provider the ``model_selection.catalog`` settings select.

    Args:
        config: The catalog settings.
        shipped_files: The provider for the files shipped in the image, used for ``file`` storage.
    """
    if config.storage == "repo":
        if config.repo_token is None:
            # ConfigModelCatalog already rejects this; kept for the type checker.
            raise ValueError("model catalog storage 'repo' needs repo_token")
        return RepoCatalogProvider(
            gitlab_url=config.repo_gitlab_url,
            project=config.repo_project,
            ref=config.repo_ref or None,
            path=config.repo_path,
            token=config.repo_token,
            timeout=config.repo_timeout,
        )
    return shipped_files
