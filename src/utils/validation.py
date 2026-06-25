"""Input validation utilities for sentence completion questions."""

from typing import List, Optional, Tuple

from src.data.prepare import CompletionQuestion
from src.utils.logger import get_logger

logger = get_logger(__name__)

BLANK_TOKENS = ["_____", "___", "__", ".....", "[ ]", "( )", "＜ ＞", "_____"]


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_question(question: str, blank_token: str = "_____") -> Tuple[bool, Optional[str]]:
    """
    Validate a question string.

    Returns:
        (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question is empty."

    if len(question.strip()) < 5:
        return False, f"Question too short: '{question}'"

    if len(question) > 2000:
        return False, "Question too long (max 2000 characters)."

    if blank_token not in question:
        return False, f"Blank token '{blank_token}' not found in question."

    return True, None


def validate_options(options: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of options.

    Returns:
        (is_valid, error_message)
    """
    if not options:
        return False, "No options provided."

    if len(options) < 2:
        return False, "At least 2 options required."

    if len(options) > 10:
        return False, "Too many options (max 10)."

    for i, opt in enumerate(options):
        if not opt or not opt.strip():
            return False, f"Option {i + 1} is empty."
        if len(opt) > 200:
            return False, f"Option {i + 1} too long (max 200 characters)."

    if len(set(options)) != len(options):
        return False, "Duplicate options detected."

    return True, None


def validate_completion_question(q: CompletionQuestion) -> Tuple[bool, Optional[str]]:
    """
    Validate a CompletionQuestion object.

    Returns:
        (is_valid, error_message)
    """
    q_valid, q_err = validate_question(q.question, q.blank_token)
    if not q_valid:
        return False, f"Question validation failed: {q_err}"

    o_valid, o_err = validate_options(q.options)
    if not o_valid:
        return False, f"Options validation failed: {o_err}"

    if q.correct is not None and q.correct not in q.options:
        return False, f"Correct answer '{q.correct}' not in options {q.options}."

    return True, None


def sanitize_question_text(text: str) -> str:
    """Clean up question text: normalize whitespace, fix common encoding issues."""
    import re
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u200b", "")   # Zero-width space
    text = text.replace("\ufeff", "")   # BOM
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_option_text(text: str) -> str:
    """Clean up option text."""
    import re
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".")
    return text
