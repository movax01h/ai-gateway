import pytest

from lib.feature_flags.context import FeatureFlag, current_feature_flag_context


@pytest.fixture(name="workspace_agents_flag")
def workspace_agents_flag_fixture():
    """Enable ``dap_workspace_agents`` for one test, then restore the previous context.

    Binding workspace agents is gated on the flag, so a test expecting items to reach the graph requests this.
    """
    token = current_feature_flag_context.set(
        current_feature_flag_context.get() | {FeatureFlag.DAP_WORKSPACE_AGENTS.value}
    )
    yield
    current_feature_flag_context.reset(token)
