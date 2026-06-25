"""Dataset loading and processing for ESL sentence completion."""

import json
import os
from typing import List, Dict, Optional

import pandas as pd
from torch.utils.data import Dataset

from src.data.prepare import CompletionQuestion, parse_questions_from_dict


class CompletionDataset:
    """Loads and manages a dataset of sentence completion questions."""

    def __init__(self, questions: List[CompletionQuestion]):
        self.questions = questions

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> CompletionQuestion:
        return self.questions[idx]

    def __iter__(self):
        return iter(self.questions)

    @classmethod
    def from_json(cls, path: str) -> "CompletionDataset":
        """Load dataset from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
        questions = parse_questions_from_dict(data)
        return cls(questions)

    @classmethod
    def from_csv(cls, path: str) -> "CompletionDataset":
        """Load dataset from a CSV file.

        Expected columns: question, options (semicolon-separated), answer
        """
        df = pd.read_csv(path)
        questions = []
        for _, row in df.iterrows():
            options = [o.strip() for o in str(row["options"]).split(";")]
            q = CompletionQuestion(
                question=row["question"],
                options=options,
                correct=row.get("answer"),
            )
            questions.append(q)
        return cls(questions)

    @classmethod
    def from_list(cls, data: List[Dict]) -> "CompletionDataset":
        """Create dataset from a list of dictionaries."""
        questions = parse_questions_from_dict(data)
        return cls(questions)

    def to_dict_list(self) -> List[Dict]:
        """Convert dataset to a list of dictionaries."""
        result = []
        for q in self.questions:
            item = {"question": q.question, "options": q.options}
            if q.correct is not None:
                item["answer"] = q.correct
            result.append(item)
        return result

    def save_json(self, path: str) -> None:
        """Save dataset to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict_list(), f, indent=2, ensure_ascii=False)

    def filter_answered(self) -> "CompletionDataset":
        """Return a new dataset with only questions that have a known answer."""
        return CompletionDataset([q for q in self.questions if q.correct is not None])


class TorchCompletionDataset(Dataset):
    """PyTorch Dataset wrapper for fine-tuning."""

    def __init__(self, questions: List[CompletionQuestion], tokenizer,
                 max_length: int = 512, blank_token: str = "_____"):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.blank_token = blank_token

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict:
        q = self.questions[idx]
        correct_idx = q.options.index(q.correct) if q.correct else 0

        # Create all option sentences
        sentences = [q.fill_blank(opt) for opt in q.options]

        return {
            "sentences": sentences,
            "correct_idx": correct_idx,
            "question": q.question,
            "options": q.options,
        }
