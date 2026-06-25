"""Data preparation utilities for sentence completion questions."""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CompletionQuestion:
    """Represents a single sentence completion question."""
    question: str
    options: List[str]
    correct: str = None
    blank_token: str = "_____"

    def fill_blank(self, option: str) -> str:
        """Fill the blank in the question with the given option."""
        return self.question.replace(self.blank_token, option)

    def get_filled_sentences(self) -> List[str]:
        """Get all possible filled sentences for each option."""
        return [self.fill_blank(opt) for opt in self.options]


def prepare_data(question: str, options: List[str],
                 blank_token: str = "_____") -> List[str]:
    """
    Prepare input data by generating complete sentences from the question
    and each option.

    Args:
        question: The sentence with blanks marked by blank_token.
        options: A list of possible options to fill in the blank.
        blank_token: The token used to mark blanks in the question.

    Returns:
        A list of processed sentences, one per option.
    """
    return [question.replace(blank_token, opt) for opt in options]


def parse_questions_from_dict(data: List[Dict]) -> List[CompletionQuestion]:
    """
    Parse a list of dictionaries into CompletionQuestion objects.

    Expected format per item:
        {"question": "...", "options": [...], "answer": "..."}
    """
    questions = []
    for item in data:
        q = CompletionQuestion(
            question=item["question"],
            options=item["options"],
            correct=item.get("answer"),
            blank_token=item.get("blank_token", "_____"),
        )
        questions.append(q)
    return questions


def split_context_and_answer(question: str, blank_token: str = "_____") -> Tuple[str, str]:
    """
    Split a question into the context before the blank and after the blank.

    Returns:
        (before_blank, after_blank)
    """
    parts = question.split(blank_token, 1)
    before = parts[0]
    after = parts[1] if len(parts) > 1 else ""
    return before, after
