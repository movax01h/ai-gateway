#!/usr/bin/env python3
"""Parse GitLab documentation markdown files into JSONL for embedding ingestion."""

import hashlib
import json
import os
import re

MAX_CHARS_PER_EMBEDDING = 1500
MIN_CHARS_PER_EMBEDDING = 100

METADATA_REGEX = re.compile(r"\A---(?P<metadata>.*?)---", re.DOTALL)


def _title(content: str) -> str | None:
    """Extract the first markdown heading as the document title."""
    match = re.search(r"#+\s+(?P<title>.+)\n", content)
    if not match:
        return None
    title = match.group("title")
    # Strip bold annotations like **(EE)** at the end
    title = re.sub(r"\*\*\(.+\)\*\*$", "", title).strip()
    return title or None


def _parse_content_and_metadata(
    content: str, md5sum: str, source_name: str, url: str
) -> tuple[str, dict]:
    """Strip YAML front matter and build metadata dict."""
    match = METADATA_REGEX.match(content)
    if match:
        content = content[match.end() :].strip()

    metadata = {
        "title": _title(content),
        "md5sum": md5sum,
        "source": source_name,
        "source_type": "doc",
        "source_url": url,
    }
    return content, metadata


def _split_by_newline_positions(content: str) -> list[str]:
    """Split content into chunks bounded by newlines, respecting min/max char limits.

    Mirrors the Ruby ``split_by_newline_positions`` logic exactly:

    1. If the whole content fits within max and is at least min chars, yield it as-is.
    2. Walk newline positions greedily, emitting chunks that are >= min chars.
    3. Any remaining tail is hard-sliced into max-char chunks (skipping slices < min chars).
    """
    chunks: list[str] = []

    if MIN_CHARS_PER_EMBEDDING <= len(content) < MAX_CHARS_PER_EMBEDDING:
        return [content]

    # Collect all newline positions
    positions = [m.start() for m in re.finditer(r"\n", content)]

    cursor = 0
    while True:
        # Find the furthest newline that keeps the chunk within max_chars
        candidates = [
            p for p in positions if cursor < p <= cursor + MAX_CHARS_PER_EMBEDDING
        ]
        if not candidates:
            break
        position = max(candidates)

        chunk = content[cursor:position]
        if len(chunk) < MIN_CHARS_PER_EMBEDDING:
            cursor = position + 1
            continue

        chunks.append(chunk)
        cursor = position + 1

    # Handle remaining tail with hard slicing
    while cursor < len(content):
        tail = content[cursor:]
        for i in range(0, len(tail), MAX_CHARS_PER_EMBEDDING):
            slice_text = tail[i : i + MAX_CHARS_PER_EMBEDDING]
            if len(slice_text) < MIN_CHARS_PER_EMBEDDING:
                cursor = len(content)
                break
            chunks.append(slice_text)
            cursor += len(slice_text)
        else:
            break

    return chunks


def parse_and_split(content: str, source_name: str, url: str) -> list[dict]:
    """Parse a single markdown document and split it into embedding-sized chunks."""
    md5sum = hashlib.sha256(content.encode()).hexdigest()
    content, metadata = _parse_content_and_metadata(content, md5sum, source_name, url)

    items = []
    for text in _split_by_newline_positions(content):
        if text is None:
            continue
        if not re.search(r"\w", text):
            continue
        items.append({"content": text, "metadata": metadata})
    return items


def parse_files(filenames: list[str], doc_dir: str, root_url: str) -> list[dict]:
    """Parse a list of markdown files and return all chunks."""
    entries = []
    for filename in filenames:
        # filename: /tmp/gitlab-docs-v17.0.1-ee/doc/user/...foo.md
        # source:   user/...foo.md
        # url:      https://docs.gitlab.com/ee/user/...foo
        source = filename.replace(f"{doc_dir}/doc/", "", 1)
        page = source.replace(".md", "")
        url = f"{root_url}/{page}"

        print(f"parsing: {filename}")

        with open(filename, "r", encoding="utf-8") as fh:
            raw = fh.read()

        entries.extend(parse_and_split(raw, source, url))
    return entries


def export(entries: list[dict], export_path: str) -> None:
    """Write entries as JSONL to export_path, replacing any existing file."""
    if os.path.exists(export_path):
        os.remove(export_path)
    with open(export_path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(
                json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n"
            )


def _glob_ruby_order(base_dir: str) -> list[str]:
    """Return .md files under base_dir in the same order as Ruby's Dir.glob('**/*.md').

    Ruby's Dir.glob traverses directory entries in sorted order but recurses into
    a subdirectory immediately when it is encountered, before processing sibling
    files. This means subdirectory contents appear before any same-prefix sibling
    files (e.g. ``analytics/index.md`` before ``analytics.md``).
    """
    result: list[str] = []

    def _visit(path: str) -> None:
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                _visit(full)
            elif entry.endswith(".md"):
                result.append(full)

    _visit(base_dir)
    return result


def main() -> None:
    """Entry point: read env vars, parse all docs, export JSONL."""
    # pylint: disable=direct-environment-variable-reference
    doc_dir = os.environ["GITLAB_DOCS_CLONE_DIR"]
    root_url = os.environ["GITLAB_DOCS_WEB_ROOT_URL"]
    export_path = os.environ["GITLAB_DOCS_JSONL_EXPORT_PATH"]
    # pylint: enable=direct-environment-variable-reference

    print(f"clone dir: {doc_dir}")

    filenames = _glob_ruby_order(f"{doc_dir}/doc")
    entries = parse_files(filenames, doc_dir, root_url)
    export(entries, export_path)


if __name__ == "__main__":
    main()
