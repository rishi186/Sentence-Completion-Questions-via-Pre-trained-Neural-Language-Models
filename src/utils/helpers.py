"""Utility helper functions."""

import json
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Get the appropriate compute device."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_results(question: str, options: List[str], scores: List[float],
                   best_option: str, correct: str = None) -> Dict[str, Any]:
    """Format prediction results into a structured dictionary."""
    result = {
        "question": question,
        "options": options,
        "scores": [round(s, 6) for s in scores],
        "predicted": best_option,
    }
    if correct is not None:
        result["correct"] = correct
        result["is_correct"] = best_option == correct
    return result
