"""Score a goal against routing keywords with BM25.

Each keyword phrase is treated as a tiny document and the goal as the search query:

    score = sum over goal words found in the keyword of
            rarity(word) * count * (k1 + 1) / (count + k1 * (1 - b + b * length / average_length))

A keyword matches when the goal scores at least `MATCH_RATIO` of what the keyword scores against
itself. A one-word keyword matches whenever it appears; half of a two-word phrase does not.
"""

import math
import re
from collections import Counter

MATCH_RATIO = 0.75

_LINKS_AND_MENTIONS = re.compile(r"https?://\S+|@\w[\w.-]*")


def words(text: str) -> list[str]:
    """Lowercase words with links and @mentions removed, and plain endings cut so forms match."""
    result = []
    for word in re.findall(r"[a-z]+", _LINKS_AND_MENTIONS.sub(" ", text.lower())):
        for ending in ("ing", "ed", "es", "s"):
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                word = word[: -len(ending)]
                break
        result.append(word[:-1] if word.endswith("e") and len(word) > 4 else word)
    return result


class KeywordBM25:
    """Match a goal against a fixed list of keyword phrases.

    Args:
        keywords: The keyword phrases, one BM25 document each.
        k1: How fast a repeated word stops adding score.
        b: How much a long phrase is discounted.

    Raises:
        ValueError: If `keywords` is empty, since BM25 needs a corpus to score against.
    """

    def __init__(self, keywords: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not keywords:
            raise ValueError("KeywordBM25 needs at least one keyword")
        self.keywords = [words(keyword) for keyword in keywords]
        self.k1, self.b = k1, b
        self.word_counts = [Counter(keyword) for keyword in self.keywords]
        self.average_length = sum(map(len, self.keywords)) / len(self.keywords)
        # A word found in few keywords says more than one found in many, so it scores higher.
        keywords_containing = Counter(
            w for keyword in self.keywords for w in set(keyword)
        )
        total = len(self.keywords)
        self.rarity = {
            w: math.log((total - n + 0.5) / (n + 0.5) + 1)
            for w, n in keywords_containing.items()
        }
        # What each keyword scores against itself: the most the goal can reach for it.
        self.full_scores = [
            self._score(keyword, i) for i, keyword in enumerate(self.keywords)
        ]

    def _score(self, goal_words: list[str], index: int) -> float:
        keyword, counts = self.keywords[index], self.word_counts[index]
        length_discount = self.k1 * (
            1 - self.b + self.b * len(keyword) / self.average_length
        )
        return sum(
            self.rarity[w] * counts[w] * (self.k1 + 1) / (counts[w] + length_discount)
            for w in set(goal_words)
            if counts[w]
        )

    def matches(self, goal: str) -> list[bool]:
        """One flag per keyword: whether the goal matches it strongly enough."""
        goal_words = words(goal)
        return [
            full > 0 and self._score(goal_words, i) >= MATCH_RATIO * full
            for i, full in enumerate(self.full_scores)
        ]
