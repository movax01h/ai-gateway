# Indexes gitlab documentation to create embeddable search for self-hosted
# AIGW instances. This was initially added to GitLab-rails repository, but it
# is only used when building the docker image here.

import argparse
import json
import logging
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from shutil import rmtree
from typing import Any, Optional
from zipfile import ZipFile

import requests
from langchain_community.docstore.document import Document
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(-\w+)?$")


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate GitLab docs index.")

    parser.add_argument("-o", "--output_path", help="Output path", default="docs.db")
    parser.add_argument(
        "-v",
        "--version_tag",
        help="GitLab version tag to include in the URL (e.g., v17.1.0-ee)",
        default="master",
    )
    return parser.parse_args()


def execution_error(error_message: str):
    logger.error(error_message)
    sys.exit(1)


def candidate_version_tags(version_tag: str) -> list[str]:
    """Return the requested tag followed by earlier patches in the same minor.

    When a patch tag (e.g. ``v19.0.1-ee``) is requested but not yet published
    on gitlab-org/gitlab, earlier patches within the same minor are safe
    substitutes because docs rarely change across patches. Non-versioned
    inputs (e.g. ``master``) are returned unchanged with no fallback list.

    Args:
        version_tag: A GitLab version tag such as ``v19.0.1-ee``, or a
            branch name.

    Returns:
        A list of candidate tags ordered from the requested patch down to
        patch 0, or a single-element list when the tag is not versioned.
    """
    match = VERSION_TAG_RE.match(version_tag)
    if not match:
        return [version_tag]
    major, minor, patch = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )
    suffix = match.group(4) or ""
    return [f"v{major}.{minor}.{p}{suffix}" for p in range(patch, -1, -1)]


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=3, min=3, max=30),
    retry=retry_if_exception_type(HTTPError),
)
def _download_docs_archive(version_tag: str) -> Optional[bytes]:
    docs_url = f"https://gitlab.com/gitlab-org/gitlab/-/archive/{version_tag}/gitlab-{version_tag}.zip?path=doc"
    logger.info("Fetching documents from %s", docs_url)
    response = requests.get(docs_url, timeout=100)
    if response.status_code == 200:
        return response.content
    if response.status_code == 429:
        response.raise_for_status()
    logger.warning(
        "Failed to download documents for %s. Status code: %d",
        version_tag,
        response.status_code,
    )
    return None


def fetch_documents(version_tag: str) -> Path:
    """Download and extract the GitLab docs archive for ``version_tag``.

    Tries the requested tag first, then earlier patch tags within the same
    minor via :func:`candidate_version_tags`. Exits the process via
    :func:`execution_error` if no candidate returns HTTP 200.

    Args:
        version_tag: A GitLab version tag such as ``v19.0.1-ee``, or
            ``master``.

    Returns:
        Path to the directory containing the extracted documentation files.
    """
    content: Optional[bytes] = None
    for candidate in candidate_version_tags(version_tag):
        try:
            content = _download_docs_archive(candidate)
        except HTTPError:
            logger.warning(
                "Rate-limited for %s after all retries; trying next candidate.",
                candidate,
            )
            content = None
        if content is not None:
            if candidate != version_tag:
                logger.warning(
                    "Docs for %s were unavailable; falling back to %s.",
                    version_tag,
                    candidate,
                )
            break

    if content is None:
        return execution_error(
            f"Failed to download documents for {version_tag} and all fallback tags."
        )

    tmpdirname = tempfile.mkdtemp()
    zip_path = Path(tmpdirname) / "docs.zip"

    with open(zip_path, "wb") as f:
        f.write(content)
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdirname)

    # Find the directory that was extracted
    extracted_dirs = [path for path in Path(tmpdirname).iterdir() if path.is_dir()]
    if not extracted_dirs:
        execution_error("No directory found after extraction. Exiting.")
    zip_path.unlink()
    logger.info("Documents are fetched.")
    extracted_dir = extracted_dirs[0]
    logger.info("Extracted documents to %s", extracted_dir)
    return extracted_dir


def build_row_corpus(row: dict):
    corpus = row["content"]
    # Remove the preamble
    preamble_start = corpus.find("---")
    if preamble_start != -1:
        preamble_end = corpus.find("---", preamble_start + 1)
        corpus = corpus[preamble_end + 2 : -1]
    if not corpus:
        return ""

    # Stemming could be helpful, but it is already applied by the sqlite
    # Remove punctuation and set to lowercase, this should reduce the size of the corpus and allow
    # the query to be a bit more robust
    corpus = corpus.lower()
    corpus = re.sub(r"[^\w\s]", "", corpus)
    return corpus


def process_file_to_db_entry(path: Path, file: Any) -> tuple:
    with file.open("r") as f:
        file_name = f.name.removeprefix(f"{path}/doc")
        doc = Document(page_content=f.read(), metadata={"filename": file_name})
        row = {"content": doc.page_content, "metadata": doc.metadata}

        return build_row_corpus(row), row["content"], json.dumps(row["metadata"])


def read_documents(path: Path):
    logger.info("Processing documents")
    doc_files = path.glob("doc/**/*.md")
    if not doc_files:
        execution_error("No markdown files found")

    return doc_files


def create_database(output_path: str, sql_tuples: list):
    logger.info("Creating database at %s", output_path)

    Path(output_path).unlink(True)

    # Create the database
    with closing(sqlite3.connect(output_path)) as connection:
        c = connection.cursor()
        c.execute(
            "CREATE VIRTUAL TABLE doc_index USING fts5(processed, content, metadata, tokenize='porter trigram');"
        )
        c.execute("PRAGMA user_version = 1;")
        c.executemany(
            "INSERT INTO doc_index (processed, content, metadata) VALUES (?,?,?)",
            sql_tuples,
        )
        connection.commit()


def build_indexed_docs():
    args = parse_arguments()

    docs_path = fetch_documents(version_tag=args.version_tag)

    if not docs_path:
        execution_error("Fetching documents failed")
    output_path = args.output_path
    files = read_documents(docs_path)
    sql_tuples = [process_file_to_db_entry(docs_path, file) for file in files]
    create_database(output_path, sql_tuples)
    rmtree(docs_path)
    logger.info("Database created at %s", output_path)
