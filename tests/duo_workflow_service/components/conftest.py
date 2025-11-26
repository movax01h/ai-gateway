import pytest


@pytest.fixture(name="graph_config")
def graph_config_fixture():
    return {"configurable": {"thread_id": "test-workflow"}}
