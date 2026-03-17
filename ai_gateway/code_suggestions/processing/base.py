from prometheus_client import Counter

LANGUAGE_COUNTER = Counter(
    "code_suggestions_prompt_language",
    "Language count by number",
    ["lang", "extension", "editor_lang"],
)
