"""
Level 3: pull a number out of retrieved text and do the arithmetic in Python
instead of trusting the LLM to calculate correctly.
"""

import re

from ingestion import Chunk

# Matches figures like "450 crore", "Rs. 1,234.5 million", "$540,000", "20%"
_NUMBER_PATTERN = re.compile(
    r"(?:rs\.?|inr|\$|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crore|million|billion|lakh)?",
    re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "lakh": 100_000,
    "crore": 10_000_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _looks_like_a_year(raw_value: str, unit: str | None) -> bool:
    """A bare 4-digit number like "2024" (no currency/unit) is almost always
    a year mention (e.g. "FY2024"), not a financial figure."""
    digits_only = raw_value.replace(",", "")
    return unit is None and digits_only.isdigit() and len(digits_only) == 4 and 1900 <= int(digits_only) <= 2100


def extract_numbers(text: str) -> list[float]:
    """Return every plausible financial figure found in a chunk of text."""
    numbers = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw_value, unit = match.groups()
        if not raw_value or _looks_like_a_year(raw_value, unit):
            continue
        value = float(raw_value.replace(",", ""))
        if unit:
            value *= _UNIT_MULTIPLIERS[unit.lower()]
        numbers.append(value)
    return numbers


def extract_number_near_keyword(text: str, keyword: str, window: int = 60) -> float | None:
    """Find `keyword` in the text and return the first number right after it."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return None
    snippet = text[idx: idx + len(keyword) + window]
    numbers = extract_numbers(snippet)
    return numbers[0] if numbers else None


def extract_primary_number(chunk: Chunk, keyword: str | None = None) -> float | None:
    """
    If `keyword` is given (e.g. "revenue", "profit"), return the number found
    right after that word — this matters because a chunk often mentions
    several figures (revenue, profit, etc.) and we need the right one.
    Falls back to "largest number in the chunk" when no keyword is given or
    the keyword isn't found. Good enough for this project's scope; not a
    general-purpose financial parser.
    """
    if keyword:
        near_keyword = extract_number_near_keyword(chunk.text, keyword)
        if near_keyword is not None:
            return near_keyword
    numbers = extract_numbers(chunk.text)
    return max(numbers) if numbers else None


def growth_percent(old_value: float, new_value: float) -> float:
    if old_value == 0:
        raise ValueError("Cannot compute growth percent when the base value is 0.")
    return ((new_value - old_value) / old_value) * 100


def difference(a: float, b: float) -> float:
    return a - b


def ratio(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot compute a ratio with a zero denominator.")
    return a / b


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


if __name__ == "__main__":
    sample = "Total revenue for FY2024 was Rs. 540 crore, up from FY2023."
    print(extract_numbers(sample))
    print(f"Growth from 450 to 540: {growth_percent(450, 540):.2f}%")
