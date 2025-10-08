"""Tests for emoji security functions."""

import json

from duo_workflow_service.security.emoji_security import strip_emojis


class TestEmojiSecurity:
    """Test suite for emoji stripping functionality."""

    def test_basic_emoji_stripping(self):
        """Test basic emoji removal from text."""
        test_input = "Hello 😀 World 🌍!"
        result = strip_emojis(test_input)
        assert "😀" not in result and "🌍" not in result
        assert "Hello" in result and "World" in result

    def test_json_encoded_emojis(self):
        """Test JSON-encoded emoji removal (surrogate pairs)."""
        test_input = json.dumps("Hello 😀 World 🌍!")
        result = strip_emojis(test_input)
        assert "\\ud83d\\ude00" not in result
        assert "\\ud83c\\udf0d" not in result
        assert "Hello" in result and "World" in result

    def test_mixed_json_and_raw_emojis(self):
        """Test mixture of JSON-encoded and raw emojis."""
        test_input = "Some text with \\ud83d\\ude00 and raw 🌍 emoji"
        result = strip_emojis(test_input)
        assert "\\ud83d\\ude00" not in result
        assert "🌍" not in result
        assert "Some text with" in result and "emoji" in result

    def test_json_encoded_with_whitespace_evasion(self):
        """Test JSON-encoded emojis with whitespace evasion."""
        test_input = "Hello \\ud83d \\ude00 World \\ud83c \\udf0d"
        result = strip_emojis(test_input)
        assert "\\ud83d" not in result and "\\ude00" not in result
        assert "\\ud83c" not in result and "\\udf0d" not in result
        assert "Hello" in result and "World" in result

    def test_common_emoji_json_patterns(self):
        """Test common emoji JSON patterns."""
        test_cases = [
            ("Smiley \\ud83d\\ude00 face", "Smiley  face"),
            ("Food \\ud83c\\udf55 time", "Food  time"),
            ("Person \\ud83d\\udc64 here", "Person  here"),
            ("Object \\ud83d\\udd0d test", "Object  test"),
        ]

        for input_text, expected_partial in test_cases:
            result = strip_emojis(input_text)
            assert "\\ud83" not in result
            assert expected_partial.replace("  ", " ").strip() in result.strip()

    def test_zero_width_character_evasion(self):
        """Test zero-width character evasion techniques."""
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
        for char in zero_width_chars:
            test_input = f"Text{char}😀{char}more"
            result = strip_emojis(test_input)
            assert "😀" not in result
            assert char not in result
            assert "Textmore" in result or "Text more" in result

    def test_skin_tone_modifiers(self):
        """Test skin tone modifier removal."""
        test_input = "Person 👋🏻👋🏽👋🏿 waving"
        result = strip_emojis(test_input)
        assert "👋" not in result
        assert "\U0001f3fb" not in result
        assert "\U0001f3fd" not in result
        assert "\U0001f3ff" not in result
        assert "Person" in result and "waving" in result

    def test_comprehensive_emoji_ranges(self):
        """Test emojis from different Unicode ranges."""
        test_cases = [
            ("Emoticons 😀😃😄", "emoticons"),
            ("Symbols 🌍🌎🌏", "symbols"),
            ("Transport 🚗🚕🚙", "transport"),
            ("Objects 📱💻⌚", "objects"),
            ("Flags 🇺🇸🇫🇷🇯🇵", "flags"),
        ]

        for input_text, expected_word in test_cases:
            result = strip_emojis(input_text).lower()
            assert expected_word in result
            assert not any(ord(c) >= 0x1F600 and ord(c) <= 0x1F6FF for c in result)

    def test_complex_mixed_content(self):
        """Test complex scenarios with mixed emoji and content."""
        test_input = """
        Meeting agenda 📅:
        1. Review 📊 reports
        2. Discuss 💬 project status
        3. Plan 🗓️ next steps

        Party tonight 🎉! Food 🍕, drinks 🍺, and music 🎵!
        """

        result = strip_emojis(test_input)

        assert "Meeting agenda" in result
        assert "Review" in result and "reports" in result
        assert "Discuss" in result and "project status" in result
        assert "Plan" in result and "next steps" in result
        assert "Party tonight" in result
        assert "Food" in result and "drinks" in result and "music" in result

        emoji_chars = ["📅", "📊", "💬", "🗓️", "🎉", "🍕", "🍺", "🎵"]
        for emoji in emoji_chars:
            assert emoji not in result

    def test_nested_dict_structure(self):
        """Test emoji stripping in nested dictionaries."""
        input_data = {
            "message": "Hello 😀 World!",
            "nested": {
                "content": "Test 🌍 message",
                "emojis": ["😊", "🎉", "clean text"],
            },
        }
        result = strip_emojis(input_data)

        assert isinstance(result, list)
        processed_data = result[0]

        assert "😀" not in processed_data["message"]
        assert (
            "Hello" in processed_data["message"]
            and "World" in processed_data["message"]
        )
        assert "🌍" not in processed_data["nested"]["content"]
        assert (
            "Test" in processed_data["nested"]["content"]
            and "message" in processed_data["nested"]["content"]
        )

        assert "😊" not in str(processed_data["nested"]["emojis"])
        assert "🎉" not in str(processed_data["nested"]["emojis"])
        assert "clean text" in str(processed_data["nested"]["emojis"])

    def test_list_processing(self):
        """Test emoji stripping in lists."""
        input_data = [
            "Hello 😀!",
            {"msg": "World 🌍!"},
            "Clean text",
            ["Nested 🎉", "More 😊"],
        ]
        result = strip_emojis(input_data)

        result_str = str(result)
        assert "😀" not in result_str
        assert "🌍" not in result_str
        assert "🎉" not in result_str
        assert "😊" not in result_str
        assert "Hello" in result_str and "World" in result_str
        assert "Clean text" in result_str
        assert "Nested" in result_str and "More" in result_str

    def test_empty_and_none_inputs(self):
        """Test handling of edge case inputs."""
        assert strip_emojis("") == ""

        result = strip_emojis(None)
        assert result is None

        assert strip_emojis({}) == [{}]
        assert strip_emojis([]) == []

    def test_whitespace_cleanup(self):
        """Test proper whitespace cleanup after emoji removal."""
        test_cases = [
            ("Text   😀   more", "Text more"),  # Multiple spaces
            ("Start😀middle😊end", "Startmiddleend"),  # No spaces around emojis
            ("Line1\n😀\n\nLine2", "Line1\nLine2"),  # Newlines
            ("  😀  Padded  😊  ", "Padded"),  # Leading/trailing spaces
        ]

        for input_text, expected_pattern in test_cases:
            result = strip_emojis(input_text)
            assert "😀" not in result and "😊" not in result
            normalized_result = " ".join(result.split())
            normalized_expected = " ".join(expected_pattern.split())
            assert normalized_expected in normalized_result

    def test_json_dumps_real_world(self):
        """Test with actual JSON.dumps() output from real scenarios."""
        test_cases = [
            {"message": "Hello 😀 User!", "status": "success 🎉"},
            ["Item 1 🔥", "Item 2 💯", "Clean item"],
            "Mixed content with 🌟 stars and 🚀 rockets",
        ]

        for case in test_cases:
            json_encoded = json.dumps(case)
            result = strip_emojis(json_encoded)

            assert "\\ud83" not in result
            assert "\\u1f" not in result.lower()
            if isinstance(case, str):
                assert "Mixed content" in result
                assert "stars" in result and "rockets" in result
            elif isinstance(case, dict):
                assert "Hello" in result and "User" in result
                assert "success" in result
            elif isinstance(case, list):
                assert "Item 1" in result and "Item 2" in result
                assert "Clean item" in result

    def test_unicode_error_handling(self):
        """Test Unicode error handling and fallback."""
        # Test with potentially problematic Unicode sequences
        test_inputs = [
            "Normal text with 😀 emoji",
            "Mixed 🌍 content",
            "",  # Edge case
        ]

        for test_input in test_inputs:
            result = strip_emojis(test_input)
            # Should not raise exceptions and should return valid string
            assert isinstance(result, str)
            # Basic validation that it's valid UTF-8
            assert result.encode("utf-8").decode("utf-8") == result

    def test_performance_with_large_text(self):
        """Test performance doesn't degrade with larger inputs."""
        # Create a large text with scattered emojis
        large_text = ""
        for i in range(100):
            large_text += f"Item {i} with content 😀 and more 🌍 text. "

        result = strip_emojis(large_text)

        # Should handle all instances
        assert "😀" not in result
        assert "🌍" not in result

        # Should preserve structure
        assert "Item 0" in result and "Item 99" in result
        assert "with content" in result and "and more" in result
