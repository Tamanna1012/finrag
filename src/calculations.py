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


def extract_numbers(text: str) -> list[float]:
    """Return every plausible financial figure found in a chunk of text."""
    numbers = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw_value, unit = match.groups()
        if not raw_value:
            continue
        value = float(raw_value.replace(",", ""))
        if unit:
            value *= _UNIT_MULTIPLIERS[unit.lower()]
        numbers.append(value)
    return numbers


def extract_primary_number(chunk: Chunk) -> float | None:
    """
    Best-effort: assume the largest number in a chunk mentioning revenue/
    profit/etc. is the actual figure (headers/years are usually smaller).
    Good enough for this project's scope; not a general-purpose parser.
    """
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
