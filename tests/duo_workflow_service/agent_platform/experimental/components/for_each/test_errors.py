from duo_workflow_service.agent_platform.experimental.components.for_each.errors import (
    ITEM_ERROR_SUBKEY,
    AllItemsFailedError,
    failed_item_errors,
    item_error_record,
)


class TestItemErrorRecord:
    def test_it_names_the_type_and_the_message(self):
        record = item_error_record(ValueError("could not parse 'a.py'"))

        assert record == {
            ITEM_ERROR_SUBKEY: {
                "type": "ValueError",
                "message": "could not parse 'a.py'",
            }
        }

    def test_it_names_the_concrete_type_rather_than_a_base(self):
        class ToolExploded(RuntimeError):
            pass

        assert item_error_record(ToolExploded("boom"))[ITEM_ERROR_SUBKEY]["type"] == (
            "ToolExploded"
        )


class TestFailedItemErrors:
    def test_it_picks_out_the_error_records(self):
        results = {
            "0": {"shout": "A"},
            "1": item_error_record(ValueError("boom")),
            "2": {"shout": "C"},
        }

        assert failed_item_errors(results) == {
            "1": {"type": "ValueError", "message": "boom"}
        }

    def test_a_components_own_error_key_is_not_a_failed_item(self):
        """``DeterministicStepComponent`` publishes an ``error`` of its own.

        A run that recorded a tool error still ran, so the entry it left is a result. This is what the namespaced sub-
        key buys.
        """
        results = {
            "0": {
                "tool_responses": None,
                "error": "tool said no",
                "execution_result": None,
            }
        }

        assert failed_item_errors(results) == {}

    def test_an_entry_that_is_not_a_dict_is_not_a_failed_item(self):
        assert failed_item_errors({"0": "whatever a body left here"}) == {}

    def test_no_results_means_no_failed_items(self):
        assert failed_item_errors({}) == {}


class TestAllItemsFailedError:
    def test_it_quotes_the_lowest_numbered_items_error(self):
        errors = {
            "10": {"type": "RuntimeError", "message": "tenth"},
            "2": {"type": "ValueError", "message": "second"},
        }

        message = str(AllItemsFailedError.for_component("review_one", errors))

        assert "item 2 raised ValueError: second" in message
        assert "tenth" not in message

    def test_it_names_the_component_and_the_item_count(self):
        errors = {
            str(index): {"type": "ValueError", "message": "x"} for index in range(3)
        }

        message = str(AllItemsFailedError.for_component("review_one", errors))

        assert "component 'review_one'" in message
        assert "all 3 of its items" in message

    def test_it_counts_the_distinct_error_types(self):
        errors = {
            "0": {"type": "ValueError", "message": "x"},
            "1": {"type": "ValueError", "message": "y"},
            "2": {"type": "RuntimeError", "message": "z"},
        }

        assert "2 distinct error type(s)" in str(
            AllItemsFailedError.for_component("review_one", errors)
        )
