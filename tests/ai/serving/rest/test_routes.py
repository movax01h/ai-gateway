import subprocess
import sys

from lib.feature_roots import default_project_root


def test_imports_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-c", "import ai.serving.rest.routes"],
        cwd=default_project_root(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
