"""Tests for lib/context/request_metadata module."""

import unittest

from lib.context.request_metadata import (
    METADATA_LABELS,
    LLMFinishReason,
    build_metadata_labels,
    client_type,
    gitlab_realm,
    gitlab_version,
    language_server_version,
)
from lib.language_server import LanguageServerVersion


class TestMetadataLabels(unittest.TestCase):
    """Tests for METADATA_LABELS constant."""

    def test_metadata_labels_contains_expected_keys(self):
        expected_keys = ["lsp_version", "gitlab_version", "client_type", "gitlab_realm"]
        self.assertEqual(METADATA_LABELS, expected_keys)

    def test_metadata_labels_is_list(self):
        self.assertIsInstance(METADATA_LABELS, list)


class TestBuildMetadataLabels(unittest.TestCase):
    """Tests for build_metadata_labels function."""

    def setUp(self):
        # Reset context vars to default before each test
        client_type.set(None)
        gitlab_realm.set(None)
        gitlab_version.set(None)
        language_server_version.set(None)

    def tearDown(self):
        # Clean up after each test
        client_type.set(None)
        gitlab_realm.set(None)
        gitlab_version.set(None)
        language_server_version.set(None)

    def test_build_metadata_labels_with_defaults(self):
        labels = build_metadata_labels()
        self.assertEqual(labels["lsp_version"], "unknown")
        self.assertEqual(labels["gitlab_version"], "unknown")
        self.assertEqual(labels["client_type"], "unknown")
        self.assertEqual(labels["gitlab_realm"], "unknown")

    def test_build_metadata_labels_with_client_type(self):
        client_type.set("vscode")
        labels = build_metadata_labels()
        self.assertEqual(labels["client_type"], "vscode")

    def test_build_metadata_labels_with_gitlab_realm(self):
        gitlab_realm.set("saas")
        labels = build_metadata_labels()
        self.assertEqual(labels["gitlab_realm"], "saas")

    def test_build_metadata_labels_with_gitlab_version(self):
        gitlab_version.set("17.0.0")
        labels = build_metadata_labels()
        self.assertEqual(labels["gitlab_version"], "17.0.0")

    def test_build_metadata_labels_with_invalid_gitlab_version(self):
        gitlab_version.set("invalid-version")
        labels = build_metadata_labels()
        self.assertEqual(labels["gitlab_version"], "unknown")

    def test_build_metadata_labels_with_language_server_version(self):
        language_server_version.set(LanguageServerVersion.from_string("8.22.0"))
        labels = build_metadata_labels()
        self.assertEqual(labels["lsp_version"], "8.22.0")

    def test_build_metadata_labels_with_all_values(self):
        client_type.set("node-grpc")
        gitlab_realm.set("self-managed")
        gitlab_version.set("17.5.0")
        language_server_version.set(LanguageServerVersion.from_string("8.30.0"))

        labels = build_metadata_labels()

        self.assertEqual(labels["client_type"], "node-grpc")
        self.assertEqual(labels["gitlab_realm"], "self-managed")
        self.assertEqual(labels["gitlab_version"], "17.5.0")
        self.assertEqual(labels["lsp_version"], "8.30.0")


class TestLLMFinishReason(unittest.TestCase):
    """Tests for LLMFinishReason enum."""

    def test_finish_reason_values(self):
        expected_values = [
            "stop",
            "end_turn",
            "length",
            "max_tokens",
            "stop_sequence",
            "tool_calls",
            "tool_use",
            "content_filter",
            "model_context_window_exceeded",
        ]
        self.assertEqual(LLMFinishReason.values(), expected_values)

    def test_finish_reason_abnormal_values(self):
        abnormal = LLMFinishReason.abnormal_values()
        self.assertIn("length", abnormal)
        self.assertIn("content_filter", abnormal)
        self.assertIn("max_tokens", abnormal)
        self.assertIn("model_context_window_exceeded", abnormal)
        self.assertEqual(len(abnormal), 4)

    def test_finish_reason_stop(self):
        self.assertEqual(LLMFinishReason.STOP.value, "stop")

    def test_finish_reason_end_turn(self):
        self.assertEqual(LLMFinishReason.END_TURN.value, "end_turn")

    def test_finish_reason_length(self):
        self.assertEqual(LLMFinishReason.LENGTH.value, "length")

    def test_finish_reason_model_context_window_exceeded(self):
        self.assertEqual(
            LLMFinishReason.MODEL_CONTEXT_WINDOW_EXCEEDED.value,
            "model_context_window_exceeded",
        )

    def test_finish_reason_is_string_enum(self):
        # LLMFinishReason inherits from str, so the value can be used as a string
        self.assertIsInstance(LLMFinishReason.STOP, str)
        self.assertEqual(LLMFinishReason.STOP.value, "stop")
        # Can compare directly with string
        self.assertEqual(LLMFinishReason.STOP, "stop")


class TestContextVariables(unittest.TestCase):
    """Tests for context variable operations."""

    def setUp(self):
        # Reset context vars to default before each test
        client_type.set(None)
        gitlab_realm.set(None)
        gitlab_version.set(None)
        language_server_version.set(None)

    def tearDown(self):
        # Clean up after each test
        client_type.set(None)
        gitlab_realm.set(None)
        gitlab_version.set(None)
        language_server_version.set(None)

    def test_client_type_default(self):
        self.assertIsNone(client_type.get())

    def test_client_type_set_and_get(self):
        client_type.set("vscode")
        self.assertEqual(client_type.get(), "vscode")

    def test_gitlab_realm_default(self):
        self.assertIsNone(gitlab_realm.get())

    def test_gitlab_realm_set_and_get(self):
        gitlab_realm.set("saas")
        self.assertEqual(gitlab_realm.get(), "saas")

    def test_gitlab_version_default(self):
        self.assertIsNone(gitlab_version.get())

    def test_gitlab_version_set_and_get(self):
        gitlab_version.set("17.0.0")
        self.assertEqual(gitlab_version.get(), "17.0.0")

    def test_language_server_version_default(self):
        self.assertIsNone(language_server_version.get())

    def test_language_server_version_set_and_get(self):
        lsp_version = LanguageServerVersion.from_string("8.22.0")
        language_server_version.set(lsp_version)
        self.assertEqual(language_server_version.get(), lsp_version)
