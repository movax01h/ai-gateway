# pylint: disable=file-naming-for-tests


def extract_location_fragment(query: str, location_type: str) -> str:
    """Extract a location inline fragment from a GraphQL query, whitespace-normalized.

    Slices from `... on <location_type>` to the fragment's matching closing brace, so
    assertions never match content outside the fragment.
    """
    marker = f"... on {location_type}"
    if marker not in query:
        raise AssertionError(f"Fragment '{marker}' not found in query")
    start = query.index(marker)
    depth = 0
    for i in range(query.index("{", start), len(query)):
        if query[i] == "{":
            depth += 1
        elif query[i] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(query[start : i + 1].split())
    raise AssertionError(f"Unbalanced braces in fragment '{marker}'")
