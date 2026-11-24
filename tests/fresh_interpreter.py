import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_in_fresh_interpreter(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter, so modules that other tests loaded do not hide an import cycle."""
    env = {
        **os.environ,  # pylint: disable=direct-environment-variable-reference
        "PYTHONPATH": str(REPO_ROOT),
        "HF_HUB_OFFLINE": "1",
        "LANGCHAIN_TRACING_V2": "false",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
