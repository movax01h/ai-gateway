import pytest

from ai_gateway.code_suggestions.processing.post.ops import clean_irrelevant_keywords

JAVA_SAMPLE_1 = """
public static void main(String[] args) {
    int number = 5;
    long factorial = calculateFactorial(number);<|cursor|>
    System.out.println("Factorial of " + number + " is: " + factorial);
}
"""

PROCESSED_JAVA_SAMPLE_1 = """
public static void main(String[] args) {
    int number = 5;
    long factorial = calculateFactorial(number);
    System.out.println("Factorial of " + number + " is: " + factorial);
}
"""


@pytest.mark.parametrize(
    ("code_sample", "expected_code"),
    [(JAVA_SAMPLE_1, PROCESSED_JAVA_SAMPLE_1)],
)
@pytest.mark.asyncio
async def test_trim_by_min_allowed_context(code_sample: str, expected_code: str):
    actual_string = clean_irrelevant_keywords(code_sample)
    assert actual_string == expected_code
