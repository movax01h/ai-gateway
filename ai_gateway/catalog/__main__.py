"""Command line for the model catalog.

Usage:
    python -m ai_gateway.catalog validate             # the catalog files shipped in this image
    python -m ai_gateway.catalog validate catalog.yml # a single-file catalog, e.g. in the catalog repository's CI

The command runs the same checks AI Gateway runs when it loads a catalog, using the code of this image. It exits
with 0 when the catalog is valid, 1 when it is not, and 2 on a usage error.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError

from ai_gateway.catalog.provider import (
    CatalogLoadError,
    CatalogProvider,
    SingleFileCatalogProvider,
)
from ai_gateway.model_selection.model_selection_config import (
    ModelSelectionConfig,
    shipped_catalog_provider,
)


def validate_catalog(provider: CatalogProvider) -> list[str]:
    """Load the catalog from ``provider`` and run every model selection check on it.

    Env overrides such as ``AIGW_MODEL_SELECTION__DEFAULT_MODELS`` are not applied: this checks the catalog data
    alone.

    Returns:
        The problems found, empty when the catalog is valid.
    """
    config = ModelSelectionConfig(default_models_override={}, catalog_provider=provider)
    try:
        config.validate()
    except (CatalogLoadError, OSError) as exc:
        return [f"Could not load the catalog: {exc}"]
    except ValidationError as exc:
        return [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
    except (KeyError, TypeError) as exc:
        return _locate_entry_errors(provider) or [f"{type(exc).__name__}: {exc}"]
    except ValueError as exc:
        return str(exc).splitlines() or [type(exc).__name__]
    return []


# The key each list entry is indexed by, which the snapshot reads before any schema check.
_ENTRY_KEYS = {
    ("models", "models"): "gitlab_identifier",
    ("unit_primitives", "configurable_unit_primitives"): "feature_setting",
}


def _locate_entry_errors(provider: CatalogProvider) -> list[str]:
    """Name the entries that are not mappings or lack their identifying key.

    Building the snapshot fails on these with a bare ``KeyError`` or ``TypeError``, which does not say which
    entry is wrong.
    """
    documents = provider.load()
    errors = []
    for (document, section), key in _ENTRY_KEYS.items():
        entries = getattr(documents, document).get(section)
        if not isinstance(entries, list):
            errors.append(f"{section}: must be a list")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{section}[{index}]: must be a mapping")
            elif key not in entry:
                errors.append(f"{section}[{index}]: missing required key '{key}'")
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_gateway.catalog")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Check a model catalog against the code of this image."
    )
    validate.add_argument(
        "catalog_file",
        nargs="?",
        type=Path,
        help="A single-file catalog. Defaults to the catalog files shipped in this image.",
    )
    args = parser.parse_args(argv)

    if args.catalog_file is not None:
        provider: CatalogProvider = SingleFileCatalogProvider(args.catalog_file)
        source = str(args.catalog_file)
    else:
        provider = shipped_catalog_provider()
        source = "the shipped catalog files"

    errors = validate_catalog(provider)
    if errors:
        print(f"Catalog in {source} is invalid:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"Catalog in {source} is valid.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
