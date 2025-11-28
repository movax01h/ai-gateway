# GraphQL Queries

This directory contains GraphQL query and mutation files used in the Duo Workflow Service.

## Structure

Each GraphQL operation is stored in a separate `.graphql` file.

## Usage

The queries are automatically loaded by the `__init__.py` module connected to the code that is using them and exported as Python constants:

```python
from duo_workflow_service.tools.work_items.queries.graphql_queries import (
    GET_GROUP_WORK_ITEM_QUERY,
    CREATE_WORK_ITEM_MUTATION,
    # ... etc
)
```

## Validation

GraphQL files in this directory are automatically validated by `scripts/validate_graphql.py` using the official `graphql-core` library to parse and validate syntax.

The validation runs:
- As part of `make lint-code` and `make check-graphql`
- Automatically on pre-commit for any `.graphql` file changes
- In the CI pipeline

## Adding New Queries

When adding a new GraphQL query or mutation:

1. Create a new `.graphql` file in this directory with a descriptive name
2. Add the query/mutation content following GraphQL syntax
3. Update `__init__.py` to load and export the new query:
   ```python
   NEW_QUERY = load_graphql_query("new_query.graphql")
   ```
4. Add the constant name to the `__all__` list in `__init__.py`
5. Run `make lint-code` to verify the syntax is correct
