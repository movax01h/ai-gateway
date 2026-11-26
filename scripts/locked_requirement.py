"""Print a pip requirement for a package in poetry.lock, with its version and file hashes.

CI jobs that install a single tool with pip use this script. They install the same version as Poetry, and pip checks the
downloaded file against the hashes in poetry.lock. For an example, see the lint:ruff CI job. The script uses only the
standard library.
"""

import sys
import tomllib
from pathlib import Path


def locked_requirement(package: str, lock_file: Path = Path("poetry.lock")) -> str:
    """Return a pip requirement with the version and the file hashes of a package in a Poetry lock file.

    Raises:
        SystemExit: The lock file does not contain the package.
    """
    with lock_file.open("rb") as file:
        packages = tomllib.load(file)["package"]

    for entry in packages:
        if entry["name"] == package:
            hashes = " ".join(f"--hash={file['hash']}" for file in entry["files"])
            return f"{package}=={entry['version']} {hashes}"

    raise SystemExit(f"{package} is not in {lock_file}")


if __name__ == "__main__":
    print(locked_requirement(sys.argv[1]))
