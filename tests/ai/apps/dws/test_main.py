import runpy
from unittest.mock import patch

import pytest

# pylint: disable=import-outside-toplevel
# ai.apps.dws.main imports block_pickle, so a module-level import would disable unpickling at collection.


def test_run_bootstrap_runs_the_server():
    from ai.apps.dws.main import run_bootstrap

    with patch("ai.apps.dws.main.run_app") as mock_run_app:
        run_bootstrap()

    mock_run_app.assert_called_once_with()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_runs_as_main_module():
    with patch("duo_workflow_service.server.run_app") as mock_run_app:
        runpy.run_module("ai.apps.dws.main", run_name="__main__")

    mock_run_app.assert_called_once_with()
