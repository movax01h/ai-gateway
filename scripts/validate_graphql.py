#!/usr/bin/env python
"""Validate GraphQL query files for syntax errors."""

import sys
from pathlib import Path

from graphql import GraphQLError, parse


def validate_graphql_syntax(content: str) -> list[str]:
    """Validate GraphQL syntax from string content.

    Args:
        content: GraphQL query/mutation/fragment string to validate

    Returns:
        List of error messages. Empty list if valid.
    """
    try:
        parse(content)
        return []
    except GraphQLError as e:
        return [str(e)]


def validate_graphql_file(file_path: Path) -> list[str]:
    """Validate a single GraphQL file and return list of errors."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return validate_graphql_syntax(content)
    except Exception as e:
        return [f"Failed to read file: {str(e)}"]


def get_graphql_directories() -> list[Path]:
    """Return list of directories to validate."""
    return [
        Path("duo_workflow_service/graphql_queries/tools_queries"),
        # Add more directories here in the future
    ]


def main() -> int:
    """Main entry point for GraphQL validation."""
    directories = get_graphql_directories()
    has_errors = False
    total_files = 0

    for graphql_dir in directories:
        if not graphql_dir.exists():
            print(
                f"Warning: GraphQL directory not found: {graphql_dir}", file=sys.stderr
            )
            continue

        graphql_files = list(graphql_dir.glob("*.graphql"))
        total_files += len(graphql_files)

        for graphql_file in sorted(graphql_files):
            errors = validate_graphql_file(graphql_file)
            if errors:
                has_errors = True
                print(f"\n{graphql_file.relative_to('.')}:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)

    if total_files == 0:
        print("Warning: No GraphQL files found in any directory", file=sys.stderr)
        return 0

    if has_errors:
        print("\n❌ GraphQL validation failed", file=sys.stderr)
        return 1

    print("✅ All GraphQL files are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
