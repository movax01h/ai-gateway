from tests.fresh_interpreter import run_in_fresh_interpreter


def test_import_ai_gateway_does_not_load_the_container():
    # A container module that imports ai_gateway submodules loops on a fresh
    # import if the package init loads ai_gateway.container.
    result = run_in_fresh_interpreter(
        "import sys, ai_gateway; assert 'ai_gateway.container' not in sys.modules"
    )

    assert result.returncode == 0, (
        "`import ai_gateway` loads ai_gateway.container. A module that the package init "
        'imports must use string markers (`Provide["pkg_models.proxy_client"]`) instead '
        "of importing ContainerApplication.\n" + result.stderr
    )
