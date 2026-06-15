"""Tests for lib/context/orbit module."""

import unittest

from lib.context.orbit import (
    build_orbit_session_summary_extras,
    init_orbit_counters,
    is_orbit_tool,
    orbit_tool_call_count,
    total_tool_call_count,
)


class TestIsOrbitTool(unittest.TestCase):
    """Tests for the strict ``orbit_`` prefix check."""

    def test_matches_valid_orbit_tool_names(self):
        self.assertTrue(is_orbit_tool("orbit_query_graph"))
        self.assertTrue(is_orbit_tool("orbit_get_graph_schema"))

    def test_rejects_non_orbit_tool_names(self):
        self.assertFalse(is_orbit_tool("read_file"))
        self.assertFalse(is_orbit_tool("git_status"))

    def test_rejects_substring_match(self):
        # "orbit" must be the prefix, not just present in the name
        self.assertFalse(is_orbit_tool("my_orbit_helper"))
        self.assertFalse(is_orbit_tool("preorbit_tool"))

    def test_rejects_bare_orbit_without_underscore(self):
        # MCP convention is "<server>_<tool>", so a tool literally named
        # "orbit" (no underscore) does not satisfy the prefix check.
        self.assertFalse(is_orbit_tool("orbit"))

    def test_rejects_empty_string(self):
        self.assertFalse(is_orbit_tool(""))

    def test_is_case_sensitive(self):
        self.assertFalse(is_orbit_tool("Orbit_query_graph"))
        self.assertFalse(is_orbit_tool("ORBIT_query_graph"))


class TestInitOrbitCounters(unittest.TestCase):
    """Tests for the counter reset helper."""

    def setUp(self):
        orbit_tool_call_count.set(0)
        total_tool_call_count.set(0)

    def tearDown(self):
        orbit_tool_call_count.set(0)
        total_tool_call_count.set(0)

    def test_resets_both_counters_from_non_zero(self):
        orbit_tool_call_count.set(5)
        total_tool_call_count.set(12)

        init_orbit_counters()

        self.assertEqual(orbit_tool_call_count.get(), 0)
        self.assertEqual(total_tool_call_count.get(), 0)

    def test_is_idempotent_when_already_zero(self):
        init_orbit_counters()
        init_orbit_counters()

        self.assertEqual(orbit_tool_call_count.get(), 0)
        self.assertEqual(total_tool_call_count.get(), 0)


class TestBuildOrbitSessionSummaryExtras(unittest.TestCase):
    """Tests for the session summary kwargs builder."""

    def setUp(self):
        orbit_tool_call_count.set(0)
        total_tool_call_count.set(0)

    def tearDown(self):
        orbit_tool_call_count.set(0)
        total_tool_call_count.set(0)

    def test_returns_none_when_no_orbit_calls(self):
        total_tool_call_count.set(7)  # non-orbit usage only

        result = build_orbit_session_summary_extras(
            workflow_id="wf-1", workflow_type="developer"
        )

        self.assertIsNone(result)

    def test_returns_full_dict_with_mixed_usage(self):
        orbit_tool_call_count.set(3)
        total_tool_call_count.set(8)

        result = build_orbit_session_summary_extras(
            workflow_id="wf-42", workflow_type="developer"
        )

        self.assertEqual(
            result,
            {
                "value": "wf-42",
                "workflow_type": "developer",
                "orbit_calls_count": 3,
                "non_orbit_tool_calls": 5,
                "total_tool_calls": 8,
            },
        )

    def test_non_orbit_calls_is_zero_when_all_calls_are_orbit(self):
        orbit_tool_call_count.set(4)
        total_tool_call_count.set(4)

        result = build_orbit_session_summary_extras(
            workflow_id="wf-only-orbit", workflow_type="chat"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["non_orbit_tool_calls"], 0)
        self.assertEqual(result["orbit_calls_count"], 4)
        self.assertEqual(result["total_tool_calls"], 4)
