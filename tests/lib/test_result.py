import pytest

from lib.result import Error, Ok, Result, ok


class TestOk:
    def test_initialization(self):
        """Test initializing Ok with different values."""
        # Test with simple types
        ok_int = Ok(123)
        assert ok_int.value == 123

        ok_str = Ok("success")
        assert ok_str.value == "success"

        # Test with complex types
        ok_list = Ok([1, 2, 3])
        assert ok_list.value == [1, 2, 3]

        ok_dict = Ok({"key": "value"})
        assert ok_dict.value == {"key": "value"}

        # Test with None
        ok_none = Ok(None)
        assert ok_none.value is None

    def test_error_property(self):
        ok_result = Ok("success")
        assert ok_result.error is None

    def test_is_ok(self):
        ok_result = Ok("success")
        assert ok_result.is_ok() is True

    def test_is_err(self):
        ok_result = Ok("success")
        assert ok_result.is_err() is False


class TestError:
    def test_initialization(self):
        # Test with standard exceptions
        error_value = ValueError("Invalid value")
        error_result = Error(error_value)
        assert error_result.error is error_value

    def test_initialization_with_custom_error(self):
        # Test with custom exception
        class CustomException(Exception):
            pass

        custom_error = CustomException("Custom error message")
        error_result = Error(custom_error)
        assert error_result.error is custom_error

    def test_value_property(self):
        error_result = Error(ValueError("test error"))
        assert error_result.value is None

    def test_is_ok(self):
        error_result = Error(ValueError("test error"))
        assert error_result.is_ok() is False

    def test_is_err(self):
        error_result = Error(ValueError("test error"))
        assert error_result.is_err() is True


class TestOkFunction:
    def test_ok_function_with_ok(self):
        """Test that ok function returns True for Ok instances."""
        result = Ok("success")
        assert ok(result) is True

    def test_ok_function_with_error(self):
        """Test that ok function returns False for Error instances."""
        result = Error(ValueError("test error"))
        assert ok(result) is False

    def test_ok_function_type_guard(self):
        """Test ok function as a type guard."""
        # The type guard functionality is primarily for static type checking
        # Here we're just testing the runtime behavior
        result: Result[str, Exception] = Ok("success")

        if ok(result):
            # If ok(result) is True, then result is narrowed to Ok[str]
            assert result.value == "success"
        else:
            pytest.fail("result should be an Ok instance")

        result_error: Result[str, Exception] = Error(ValueError("test error"))

        if not ok(result_error):
            # If not ok(result_error), then result_error is narrowed to Error[Exception]
            assert isinstance(result_error.error, ValueError)
        else:
            pytest.fail("result_error should be an Error instance")
