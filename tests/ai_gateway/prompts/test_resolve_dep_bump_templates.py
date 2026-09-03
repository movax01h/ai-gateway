# pylint: disable=file-naming-for-tests
"""Both branches of the resolve_dep_bump prompts must render.

Most of this flow's guidance is Jinja, and a swallowed `{% else %}`, an unbalanced `{% endif %}`
or a stale step number would otherwise ship silently. The fallback branch matters most: it is the
path taken whenever the `run_commands` privilege is absent or a tool policy denies it.
"""

import pytest

from ai_gateway.prompts.base import jinja_env


@pytest.mark.parametrize("run_command", [True, False])
def test_user_prompt_branches_on_run_command(run_command):
    out = jinja_env.get_template(
        "resolve_dep_bump_pipeline_fix/user/1.0.0.jinja"
    ).render(tools_enabled={"run_command": run_command, "web_search": False})

    assert ("pip download" in out) is run_command
    assert ("`run_command` is not available" in out) is not run_command
    # the procedure is renumbered around the branch, so pin the last step in both
    assert "\n13. `create_merge_request_note`" in out


@pytest.mark.parametrize("run_command", [True, False])
def test_system_prompt_branches_on_run_command(run_command):
    out = jinja_env.get_template(
        "resolve_dep_bump_pipeline_fix/system/1.0.0.jinja"
    ).render(tools_enabled={"run_command": run_command, "web_search": False})

    assert ("Introspect the dependency with `run_command`" in out) is run_command
    assert ("symbol-diff commands" in out) is run_command
