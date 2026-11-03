# pylint: disable=file-naming-for-tests
"""The Duo Developer instructions must not depend on the session.

Every prompt built on ``developer/common/system`` sends its system prompt as two system messages: the instructions, then the session context
(date, project, execution environment, AGENTS.md, workspace skills). The default cache breakpoint sits on the first
message, so a session-specific value in it would make every session write the tools and the instructions to the
prompt cache again (https://gitlab.com/gitlab-org/gitlab/-/work_items/628100).
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.base import jinja_env
from ai_gateway.prompts.feature_roots import discover_feature_prompts
from lib.version import resolve_version

_ROOT = Path(__file__).parent.parent.parent
_CODE_FEATURES = _ROOT / "ai" / "features" / "code"


def _base_prompt_system() -> list[str]:
    """``developer_agent_prompt`` as resolved by the interactive flows (``^1.0.0``)."""
    base_dir = _CODE_FEATURES / "developer_agent_prompt" / "prompts" / "base"
    version = resolve_version([p.stem for p in base_dir.glob("*.yml")], "^1.0.0")
    base = yaml.safe_load((base_dir / f"{version}.yml").read_text())
    return base["prompt_template"]["system"]


def _flow_prompt_system(config: Path, prompt_name: str):
    def load() -> list[str]:
        flow = yaml.safe_load(config.read_text())
        prompt = next(p for p in flow["prompts"] if p["name"] == prompt_name)
        return prompt["prompt_template"]["system"]

    return load


_DEVELOPER_CONFIGS = _CODE_FEATURES / "developer" / "config"
# experimental/fix_pipeline also uses the two-message layout, but its own instructions contain per-run values
# (pipeline URL, merge request URL, workflow ID), so they can't be identical across sessions.
_LAYOUTS = {
    "interactive": _base_prompt_system,
    "developer/2.0.0": _flow_prompt_system(
        _DEVELOPER_CONFIGS / "2.0.0.yml", "developer_agent_prompt"
    ),
    "developer/2.0.0-goal": _flow_prompt_system(
        _DEVELOPER_CONFIGS / "2.0.0-goal.yml", "developer_agent_prompt"
    ),
    "developer/3.0.0": _flow_prompt_system(
        _DEVELOPER_CONFIGS / "3.0.0.yml", "developer_agent_prompt"
    ),
}


_SESSIONS = [
    {
        "today": "2026-11-02",
        "project_id": 278964,
        "project_url": "https://gitlab.com/gitlab-org/gitlab.git",
        "primary_branch": "master",
        "agent_user_environment": '{"shell_name":"zsh","cwd":"/Users/alice/src/gitlab"}',
        "agents_dot_md": "Run bin/rspec for tests.",
        "workspace_agent_skills": "<location>/Users/alice/.gitlab/duo/skills/use-glab/SKILL.md</location>",
    },
    {
        "today": "2026-12-24",
        "project_id": 34675721,
        "project_url": "https://gitlab.com/gitlab-org/cli.git",
        "primary_branch": "main",
        "agent_user_environment": '{"shell_name":"bash","cwd":"/home/bob/cli"}',
        "agents_dot_md": "Use make test.",
        "workspace_agent_skills": "<location>/home/bob/.gitlab/duo/skills/use-glab/SKILL.md</location>",
    },
]


@pytest.mark.parametrize("system", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
@pytest.mark.parametrize("plan_enabled", [False, True])
def test_instructions_are_identical_across_sessions(system, plan_enabled: bool) -> None:
    discover_feature_prompts()
    instructions, session_context = zip(
        *(
            [
                jinja_env.from_string(part).render(**session, plan_enabled=plan_enabled)
                for part in system()
            ]
            for session in _SESSIONS
        )
    )

    assert instructions[0] == instructions[1]
    for session, context in zip(_SESSIONS, session_context):
        for value in session.values():
            assert str(value) not in instructions[0]
            assert str(value) in context
