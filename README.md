# AI Gateway and Agent Platform Repository

This repository contains standalone services that give access to AI features to all users of
GitLab, no matter which instance they are using: self-managed, dedicated or GitLab.com. In this repository are these distinct but related services:

1. The [AI Gateway](ai_gateway/README.md), which handles things such as prompts, model and provider selection, and custom models. This is the service named `gitlab-ai-gateway` in your GDK.
1. The [Duo Workflow Service](duo_workflow_service/README.md) (DWS), also known as the Agent Platform, which handles Agentic functionality, and makes calls to LLMs via the AI Gateway service. This is the service named `duo-workflow-service` in your GDK.

## How to become a project maintainer

See [Maintainership](docs/maintainership.md).

## Testing

See [test doc](docs/tests.md).

## Linting

This project uses the following linting tools:

- [Black](https://black.readthedocs.io/): Enforces a consistent code style.
- [isort](https://pycqa.github.io/isort/): Organizes and sorts imports.
- [pylint](https://pylint.pycqa.org): Analyzes code for potential errors and style issues.
- [mypy](https://mypy-lang.org): Performs static type checking.

To lint the entire projects, you can use the following command:

```shell
make lint
```

We are incrementally rolling out `mypy` static type checker to the project
([issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/246)).
To show outstanding `mypy` warnings, you can use the following command:

```shell
make check-mypy TODO=true
```

To fix linting errors, you can use the following command:

```shell
make format
```

The `format` command only addresses `black` and `isort` issues.

There is an [internal recording](https://youtu.be/SXfLOYm4zS4) for GitLab members that provides an overview of this project.

### Running lint on Git commit

We use [Lefthook](https://github.com/evilmartians/lefthook) to lint code and doc
prior to Git committing. This repository comes with a Lefthook configuration
(`lefthook.yml`), but it must be installed.

1. Install Lefthook managed Git hooks:

   ```shell
   lefthook install
   ```

1. Test Lefthook is working by running the Lefthook `pre-commit` Git hook:

   ```shell
   lefthook run pre-commit
   ```

   This should return the Lefthook version and the list of executable commands with
   output.

#### Disable Lefthook temporarily

To disable Lefthook temporarily, you can set the `LEFTHOOK` environment variable
to `0`. For instance:

```shell
LEFTHOOK=0 git commit ...
```

#### Run Lefthook hooks manually

To run the `pre-commit` Git hook, run:

```shell
lefthook run pre-commit
```

## Troubleshooting

### Installation of Poetry 1.8.3 fails

You might encounter a known symlink failure when installing `poetry` during `mise install`.

The error may look something like:

```shell
Error output:
dyld[87914]: Library not loaded: @executable_path/../lib/libpython3.10.dylib
  Referenced from: <4C4C4415-5555-3144-A171-523C428CAE71> /Users/yourusername/Code/ai-assist/.venv/bin/python
  Reason: tried: '/Users/yourusername/Code/ai-assist/.venv/lib/libpython3.10.dylib' (no such file)
```

To fix the issue, locate the `libpython3.10.dylib` on your system. Once you have located the file, use the `ln -s` command to create a symbolic link from the location where `poetry` expects it to be to where it is actually located.

Example command:

```shell
ln -s /Users/yourusername/.local/share/mise/installs/python/3.10.14/lib/libpython3.10.dylib /Users/yourusername/Code/ai-assist/.venv/lib/libpython3.10.dylib
```

Next, try installing `poetry` again.

### How to manually activate the virtualenv

- `poetry shell` or `poetry install` should create the virtualenv environment.
- To activate virtualenv, use command: `. ./.venv/bin/activate`.
- To deactivate your virtualenv, use command: `deactivate`.
- To list virtualenvs, use `poetry env list`.
- To remove virtualenv, use `poetry env remove [name of virtualenv]`.

### Resolving Dependency Conflicts with Poetry

```shell
poetry install --sync
```

If you're experiencing unexpected package conflicts, import errors, or your development environment has accumulated extra packages
over time, the `--sync` flag ensures your environment exactly matches the project's lock file. This command installs missing
dependencies, removes any extraneous packages that aren't defined in poetry.lock, effectively resetting your environment to a clean
state.

This is particularly useful when switching between branches with different dependencies, after removing packages from
`pyproject.toml`, or when your local environment has diverged from the project's intended state.

## Authentication

See [authentication and authorization doc](docs/auth.md).

## Internal Events

See [internal events doc](docs/internal_events.md) for more information on how
to add internal events and test internal event collection with Snowplow locally.

## Release

See [release doc](docs/release.md).

## Rate limiting

Access to AI Gateway is subjected to rate limiting defined as part of <https://gitlab.com/gitlab-com/runbooks/-/blob/master/docs/cloud_connector/README.md#rate-limiting>.
