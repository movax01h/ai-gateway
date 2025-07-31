import ast
from pathlib import Path

# editorconfig-checker-disable
HEADER_TEXT = """# Dependencies for Duo Workflow Service. Extended tests are run when these change patterns are matched.
# Note: Do not modify this file manually. Instead, run: make duo-workflow-service-dependencies
.duo-workflow-service-dependencies:
  changes:
    - duo_workflow_service/**/*
    - tests/duo_workflow_service/**/*
    - contract/**/*
    - pyproject.toml
"""
# editorconfig-checker-enable

OUTPUT_FILE_PATH = ".gitlab/ci/dws-dependencies.yml"


def main():
    target_directory = "duo_workflow_service/"

    # Get lines that reference ai_gateway
    change_patterns = []
    target_dir_path = Path(target_directory)
    search_string = "ai_gateway"
    for root, _, filenames in target_dir_path.walk():
        for filename in filenames:
            full_path = Path(root) / filename
            if full_path.suffix == ".py":
                with open(full_path, "r") as file:
                    content = file.read()
                    tree = ast.parse(content, filename=full_path)
                    for node in ast.walk(tree):
                        # Handle "import ..." statements
                        if isinstance(node, ast.Import):
                            matching = [
                                alias.name
                                for alias in node.names
                                if search_string in alias.name
                            ]
                            if len(matching) > 0:
                                import_path = convert_to_path(matching[0])
                                change_patterns.append(import_path)

                        # Handle "from ... import" statements
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module if node.module else ""
                            if module and search_string in module:
                                import_path = convert_to_path(module)
                                change_patterns.append(import_path)

    change_patterns = sorted(set(change_patterns))
    with open(OUTPUT_FILE_PATH, "w") as output_file:
        output_file.write(HEADER_TEXT)
        for pattern in change_patterns:
            output_file.write(f"    - {pattern}\n")


def convert_to_path(module_path):
    module_path = module_path.replace(".", "/")

    path = Path(module_path)
    if path.is_dir():
        return module_path + "/*.py"

    path = Path(module_path + ".py")
    if path.is_file():
        return module_path + ".py"

    raise Exception(  # pylint: disable=broad-exception-raised)
        f"Error, invalid module path: {module_path}"
    )


if __name__ == "__main__":
    main()
