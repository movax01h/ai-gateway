"""Tests for lib/context/auth module."""

import unittest
from unittest.mock import MagicMock

from lib.context.auth import StarletteUser, cloud_connector_token_context_var


class TestCloudConnectorTokenContextVar(unittest.TestCase):
    """Tests for cloud_connector_token_context_var."""

    def test_context_var_exists(self):
        # Just verify the context var is accessible
        self.assertIsNotNone(cloud_connector_token_context_var)

    def test_context_var_set_and_get(self):
        cloud_connector_token_context_var.set("test-token")
        self.assertEqual(cloud_connector_token_context_var.get(), "test-token")
        # Clean up
        cloud_connector_token_context_var.set(None)


class TestStarletteUser(unittest.TestCase):
    """Tests for StarletteUser class."""

    def setUp(self):
        # Create a mock CloudConnectorUser
        self.mock_cc_user = MagicMock()
        self.mock_cc_user.is_authenticated = True
        self.mock_cc_user.global_user_id = "user-123"
        self.mock_cc_user.claims = {"sub": "user-123"}
        self.mock_cc_user.is_debug = False
        self.mock_cc_user.unit_primitives = []

    def test_is_authenticated(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertTrue(user.is_authenticated)

    def test_is_authenticated_false(self):
        self.mock_cc_user.is_authenticated = False
        user = StarletteUser(self.mock_cc_user)
        self.assertFalse(user.is_authenticated)

    def test_global_user_id(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertEqual(user.global_user_id, "user-123")

    def test_claims(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertEqual(user.claims, {"sub": "user-123"})

    def test_is_debug(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertFalse(user.is_debug)

    def test_unit_primitives(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertEqual(user.unit_primitives, [])

    def test_can_delegates_to_cloud_connector_user(self):
        self.mock_cc_user.can.return_value = True
        user = StarletteUser(self.mock_cc_user)

        result = user.can("test_primitive")

        self.assertTrue(result)
        self.mock_cc_user.can.assert_called_once_with("test_primitive", None)

    def test_cloud_connector_token(self):
        user = StarletteUser(self.mock_cc_user, cloud_connector_token="token-abc")
        self.assertEqual(user.cloud_connector_token, "token-abc")

    def test_cloud_connector_token_default_none(self):
        user = StarletteUser(self.mock_cc_user)
        self.assertIsNone(user.cloud_connector_token)
