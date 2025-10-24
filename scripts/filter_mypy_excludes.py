"""Filter files based on mypy exclude patterns defined in the Makefile.

This script reads the mypy exclude patterns from the Makefile's MYPY_LINT_TODO_DIR
variable and removes any matching files from the provided list. This is useful
when you need to run mypy on a subset of files while respecting the project's
exclude configuration.

When mypy is run with explicit file arguments, it ignores --exclude flags and
checks all provided files. This script pre-filters the file list to honor
those exclusions.

Usage:
    filter_mypy_excludes.py <file1> [file2] [file3] ...

Output:
    Space-separated list of files that are not excluded from mypy checking
"""

import re
import sys


def main():
    if len(sys.argv) < 2:
        return  # No files to filter

    # Read Makefile and extract exclude patterns
    with open("Makefile", "r") as f:
        content = f.read()

    # Regex for finding MYPY_LINT_TODO lines
    mypy_regex_pattern = r"MYPY_LINT_TODO_DIR\s*\?=\s*(.*?)(?=\n\n|\n[A-Z_]|\Z)"
    if match := re.search(mypy_regex_pattern, content, re.MULTILINE | re.DOTALL):
        excludes = re.findall(r'--exclude "([^"]+)"', match.group(1))
    else:
        excludes = []
    files = sys.argv[1:]

    # Filter out excluded files
    filtered = [f for f in files if not any(exc in f for exc in excludes)]

    if filtered:
        print(" ".join(filtered))


if __name__ == "__main__":
    main()
