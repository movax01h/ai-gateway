"""Tests for the ``index_docs_as_sqlite`` build-time helper."""

import io
import zipfile
from unittest.mock import MagicMock

import pytest
from requests.exceptions import HTTPError
from tenacity import wait_none

from ai_gateway.scripts import index_docs_as_sqlite
from ai_gateway.scripts.index_docs_as_sqlite import (
    _download_docs_archive,
    candidate_version_tags,
    fetch_documents,
)


class TestCandidateVersionTags:
    @pytest.mark.parametrize(
        "version_tag,expected",
        [
            ("v19.0.1-ee", ["v19.0.1-ee", "v19.0.0-ee"]),
            ("v19.0.0-ee", ["v19.0.0-ee"]),
            (
                "v17.5.3-ee",
                ["v17.5.3-ee", "v17.5.2-ee", "v17.5.1-ee", "v17.5.0-ee"],
            ),
            ("v19.0.1", ["v19.0.1", "v19.0.0"]),
            ("master", ["master"]),
            ("some-branch", ["some-branch"]),
        ],
    )
    def test_enumerates_earlier_patches_for_versioned_tags(self, version_tag, expected):
        assert candidate_version_tags(version_tag) == expected


def _mock_response(status_code: int, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.content = content

    def raise_for_status():
        if 400 <= status_code < 600:
            raise HTTPError(response=response)

    response.raise_for_status = raise_for_status
    return response


class TestDownloadDocsArchive:
    def test_returns_content_on_200(self, monkeypatch):
        monkeypatch.setattr(
            index_docs_as_sqlite.requests,
            "get",
            lambda url, timeout: _mock_response(200, b"zip-bytes"),
        )
        assert _download_docs_archive("v19.0.0-ee") == b"zip-bytes"

    def test_retries_on_429(self, monkeypatch):
        call_count = 0

        def fake_get(_url, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _mock_response(429)
            return _mock_response(200, b"zip-bytes")

        monkeypatch.setattr(index_docs_as_sqlite.requests, "get", fake_get)
        monkeypatch.setattr(_download_docs_archive.retry, "wait", wait_none())

        assert _download_docs_archive("master") == b"zip-bytes"
        assert call_count == 3

    def test_raises_after_exhausting_retries_on_429(self, monkeypatch):
        call_count = 0

        def fake_get(_url, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(429)

        monkeypatch.setattr(index_docs_as_sqlite.requests, "get", fake_get)
        monkeypatch.setattr(_download_docs_archive.retry, "wait", wait_none())

        with pytest.raises(HTTPError):
            _download_docs_archive("master")
        assert call_count == 5

    def test_returns_none_on_non_200(self, monkeypatch):
        monkeypatch.setattr(
            index_docs_as_sqlite.requests,
            "get",
            lambda url, timeout: _mock_response(404),
        )
        assert _download_docs_archive("v19.0.1-ee") is None


def _docs_zip_bytes(extracted_dir: str = "gitlab-v19.0.0-ee-doc") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{extracted_dir}/doc/index.md", "# Index\n")
    return buf.getvalue()


class TestFetchDocuments:
    def test_uses_requested_tag_when_available(self, monkeypatch):
        monkeypatch.setattr(
            index_docs_as_sqlite.requests,
            "get",
            lambda url, timeout: _mock_response(200, _docs_zip_bytes()),
        )

        path = fetch_documents("v19.0.0-ee")

        assert path.is_dir()
        assert (path / "doc" / "index.md").exists()

    def test_falls_back_to_earlier_patch_when_requested_tag_missing(
        self, monkeypatch, caplog
    ):
        def fake_get(url, **_kwargs):
            if "v19.0.1-ee" in url:
                return _mock_response(404)
            if "v19.0.0-ee" in url:
                return _mock_response(200, _docs_zip_bytes())
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(index_docs_as_sqlite.requests, "get", fake_get)

        with caplog.at_level("WARNING"):
            path = fetch_documents("v19.0.1-ee")

        assert path.is_dir()
        assert any(
            "falling back to v19.0.0-ee" in record.message.lower()
            for record in caplog.records
        )

    def test_retries_429_then_falls_back_to_next_candidate(self, monkeypatch, caplog):
        def fake_get(_url, **_kwargs):
            if "v19.0.1-ee" in _url:
                return _mock_response(429)
            if "v19.0.0-ee" in _url:
                return _mock_response(200, _docs_zip_bytes())
            raise AssertionError(f"unexpected URL {_url}")

        monkeypatch.setattr(index_docs_as_sqlite.requests, "get", fake_get)
        monkeypatch.setattr(_download_docs_archive.retry, "wait", wait_none())

        with caplog.at_level("WARNING"):
            path = fetch_documents("v19.0.1-ee")

        assert path.is_dir()
        assert any(
            "rate-limited" in record.message.lower() for record in caplog.records
        )
        assert any(
            "falling back to v19.0.0-ee" in record.message.lower()
            for record in caplog.records
        )

    def test_exits_when_all_candidates_fail(self, monkeypatch):
        monkeypatch.setattr(
            index_docs_as_sqlite.requests,
            "get",
            lambda url, timeout: _mock_response(404),
        )

        with pytest.raises(SystemExit):
            fetch_documents("v19.0.1-ee")
