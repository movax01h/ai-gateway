"""Tests for GraphQL validation script."""

from scripts.validate_graphql import validate_graphql_syntax


class TestValidateGraphQL:
    """Test GraphQL validation functions."""

    def test_valid_graphql_query(self):
        """Test that valid GraphQL query passes without errors."""
        valid_query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                name
                email
            }
        }
        """
        errors = validate_graphql_syntax(valid_query)
        assert len(errors) == 0

    def test_valid_graphql_mutation(self):
        """Test that valid GraphQL mutation passes without errors."""
        valid_mutation = """
        mutation CreateUser($input: UserInput!) {
            createUser(input: $input) {
                user {
                    id
                    name
                }
                errors
            }
        }
        """
        errors = validate_graphql_syntax(valid_mutation)
        assert len(errors) == 0

    def test_unbalanced_braces(self):
        """Test detection of unbalanced braces."""
        invalid_query = """
        query GetUser {
            user {
                name
        }
        """
        errors = validate_graphql_syntax(invalid_query)
        assert len(errors) > 0
        assert any("Expected" in error or "Syntax Error" in error for error in errors)

    def test_unbalanced_brackets(self):
        """Test detection of unbalanced brackets."""
        invalid_query = """
        query GetUsers($types: [String!) {
            users(types: $types) {
                name
            }
        }
        """
        errors = validate_graphql_syntax(invalid_query)
        assert len(errors) > 0
        assert any("Expected" in error or "Syntax Error" in error for error in errors)

    def test_mismatched_delimiters(self):
        """Test detection of mismatched delimiters."""
        invalid_query = """
        query GetUser {
            user {
                name
            ]
        }
        """
        errors = validate_graphql_syntax(invalid_query)
        assert len(errors) > 0
        assert any("Expected" in error or "Syntax Error" in error for error in errors)

    def test_no_operation_definition(self):
        """Test that anonymous queries (shorthand syntax) are valid."""
        # Note: GraphQL allows anonymous queries with shorthand syntax
        query = """
        {
            user {
                name
            }
        }
        """
        errors = validate_graphql_syntax(query)
        # This is actually valid GraphQL (anonymous query)
        assert len(errors) == 0

    def test_unclosed_string(self):
        """Test detection of unclosed string literals."""
        invalid_query = """
        query GetUser {
            user(name: "John) {
                email
            }
        }
        """
        errors = validate_graphql_syntax(invalid_query)
        assert len(errors) > 0
        assert any("string" in error.lower() for error in errors)

    def test_comments_ignored(self):
        """Test that comments are properly ignored in validation."""
        query_with_comments = """
        # This is a comment with unbalanced { bracket
        query GetUser($id: ID!) {
            # Another comment with }
            user(id: $id) {
                name # inline comment with (
            }
        }
        """
        errors = validate_graphql_syntax(query_with_comments)
        assert len(errors) == 0

    def test_strings_with_delimiters_ignored(self):
        """Test that delimiters inside strings are ignored."""
        query_with_string_delimiters = """
        query GetUser {
            user(description: "This has { and } and [ and ]") {
                name
            }
        }
        """
        errors = validate_graphql_syntax(query_with_string_delimiters)
        assert len(errors) == 0

    def test_fragment_definition(self):
        """Test that fragment definitions are recognized."""
        fragment = """
        fragment UserFields on User {
            id
            name
            email
        }
        """
        errors = validate_graphql_syntax(fragment)
        assert len(errors) == 0

    def test_invalid_operation_name(self):
        """Test detection of invalid operation names."""
        invalid_query = """
        query 123InvalidName {
            user {
                name
            }
        }
        """
        errors = validate_graphql_syntax(invalid_query)
        assert len(errors) > 0
        assert any("Expected" in error or "Syntax Error" in error for error in errors)

    def test_multiple_errors(self):
        """Test that syntax errors are detected."""
        invalid_query = """
        query {
            user {
                name
            ]
        """
        errors = validate_graphql_syntax(invalid_query)
        # GraphQL parser stops at first syntax error
        assert len(errors) >= 1

    def test_extra_closing_brace(self):
        """Test detection of extra closing brace."""
        invalid_mutation = """
        mutation CreateNote($input: CreateNoteInput!) {
            createNote(input: $input) {
                note {
                    id
                }
                errors
            }
        }
        }
        """
        errors = validate_graphql_syntax(invalid_mutation)
        assert len(errors) > 0
        assert any("Expected" in error or "Syntax Error" in error for error in errors)
