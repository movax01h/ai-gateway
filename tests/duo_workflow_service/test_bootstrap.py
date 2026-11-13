import importlib
import runpy
from unittest.mock import patch

import pytest

# pylint: disable=import-outside-toplevel
# ai.apps.dws.main imports block_pickle, so a module-level import would disable unpickling at collection.


def test_deprecated_path_reexports_run_bootstrap():
    import duo_workflow_service.bootstrap
    from ai.apps.dws.main import run_bootstrap

    with pytest.warns(DeprecationWarning, match="ai.apps.dws.main"):
        deprecated = importlib.reload(duo_workflow_service.bootstrap)

    assert deprecated.run_bootstrap is run_bootstrap


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_deprecated_path_runs_as_main_module():
    with (
        patch("ai.apps.dws.main.run_bootstrap") as mock_run_bootstrap,
        pytest.warns(DeprecationWarning, match="ai.apps.dws.main"),
    ):
        runpy.run_module("duo_workflow_service.bootstrap", run_name="__main__")

    mock_run_bootstrap.assert_called_once_with()
