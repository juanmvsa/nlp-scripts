from langdetect import detect
from typing import List


def remove_english_tokens(tokens: List[str]) -> List[str]:
    """Removes all tokens detected as English from the list."""
    return [token for token in tokens if detect(token) != "en"]
