"""PDF parsing module for extracting sentence completion questions from PDF files.

Supports multiple common formats with fallback strategies:
  Format 1 (Lettered options, each on own line):
    1. The cat sat on the _____.
       A) mat
       B) table
       C) chair
       D) roof

  Format 2 (Numbered options):
    1. The cat sat on the _____.
       1) mat
       2) table

  Format 3 (Inline options):
    1. The cat sat on the _____. A) mat B) table C) chair D) roof

  Format 4 (Parenthesis options):
    1. The cat sat on the _____.
    (A) mat  (B) table  (C) chair  (D) roof

  Format 5 (Bracket options):
    1. The cat sat on the _____.
    [A] mat  [B] table  [C] chair  [D] roof

  Also supports answer keys at the end of the document:
    Answer Key: 1-A, 2-B, 3-C
    Answers: 1. A, 2. B, 3. C
    1.A 2.B 3.C
"""

import re
import io
from typing import List, Dict, Tuple, Optional

import pdfplumber

from src.data.prepare import CompletionQuestion
from src.utils.logger import get_logger
from src.utils.validation import (
    validate_question, validate_options, sanitize_question_text,
    sanitize_option_text,
)

logger = get_logger(__name__)


# --- Regex patterns ---

# Question number prefix: "1." "1)" "1:" "Q1." "Question 1."
Q_NUM_RE = re.compile(
    r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+)\s*[\.\)\:]\s*',
    re.IGNORECASE,
)

# Option patterns — try multiple formats
OPTION_LETTER_RE = re.compile(
    r'(?:^|\n|\s)\(?([A-Da-d])\)?\s*[\)\.\]\:]\s*(.+?)(?=(?:\n|\s)\(?[A-Da-d]\)?\s*[\)\.\]\:]|\Z)',
    re.DOTALL,
)

OPTION_BRACKET_RE = re.compile(
    r'(?:^|\n|\s)\[([A-Da-d])\]\s*(.+?)(?=(?:\n|\s)\[[A-Da-d]\]|\Z)',
    re.DOTALL,
)

OPTION_NUMBER_RE = re.compile(
    r'(?:^|\n|\s)(\d+)\s*[\)\.\]\:]\s*(.+?)(?=(?:\n|\s)\d+\s*[\)\.\]\:]|\Z)',
    re.DOTALL,
)

# Answer key patterns
ANSWER_KEY_HEADER_RE = re.compile(
    r'(?:answer\s*(?:key|s)?|key|solutions?)\s*[\:\-]?\s*(.+)',
    re.IGNORECASE,
)

ANSWER_ENTRY_RE = re.compile(
    r'(\d+)\s*[\.\)\-\:]\s*([A-Da-d])',
)

# Inline answer: "1.A" or "1-A" or "1)A"
ANSWER_INLINE_RE = re.compile(
    r'(\d+)\s*[\.\)\-]\s*([A-Da-d])\b',
)

# Blank patterns — various representations
BLANK_PATTERNS = [
    re.compile(r'_{2,}'),        # _____
    re.compile(r'\.{5,}'),       # .....
    re.compile(r'\*{5,}'),       # *****
    re.compile(r'\[\s*\]'),      # [ ]
    re.compile(r'\(\s*\)'),      # ( )
    re.compile(r'＜\s*＞'),      # ＜ ＞
    re.compile(r'⌀'),            # empty set symbol
]

# Multi-blank pattern (for questions with multiple blanks)
MULTI_BLANK_RE = re.compile(r'(_{2,}|\.{5,}|\*{5,}|\[\s*\]|\(\s*\))')


def normalize_blank(text: str) -> str:
    """Normalize various blank representations to _____."""
    for pattern in BLANK_PATTERNS:
        text = pattern.sub('_____', text)
    return text


def has_blank(text: str) -> bool:
    """Check if text contains any blank pattern."""
    for pattern in BLANK_PATTERNS:
        if pattern.search(text):
            return True
    return '_____' in text


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_file: A file path (str) or file-like object (BytesIO).

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If no text could be extracted.
    """
    text_parts = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text(
                        x_tolerance=2, y_tolerance=2
                    )
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    # Try simpler extraction
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception:
                        logger.error(f"Page {page_num} completely failed.")
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

    text = '\n'.join(text_parts)
    if not text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The PDF might be scanned images — OCR would be needed."
        )

    return text


def extract_answer_key(text: str) -> Dict[int, str]:
    """
    Extract answer key from the document if present.

    Returns:
        Dict mapping question number to answer letter (e.g., {1: 'A', 2: 'B'}).
    """
    answers = {}

    # Strategy 1: Look for "Answer Key:" header, then search all text after it
    for match in ANSWER_KEY_HEADER_RE.finditer(text):
        # Get all text from the header match to the end of the document
        remaining_text = text[match.start():]
        for entry_match in ANSWER_ENTRY_RE.finditer(remaining_text):
            q_num = int(entry_match.group(1))
            answer_letter = entry_match.group(2).upper()
            answers[q_num] = answer_letter

    if answers:
        return answers

    # Strategy 2: Look for a block of "1.A 2.B 3.C" at the end
    lines = text.strip().split('\n')
    in_answer_section = False
    for line in lines:
        line = line.strip()
        if re.match(r'^(answer|key|solutions?)', line, re.IGNORECASE):
            in_answer_section = True
            # Also check if answers are on the same line
            for m in ANSWER_INLINE_RE.finditer(line):
                q_num = int(m.group(1))
                answers[q_num] = m.group(2).upper()
            continue
        if in_answer_section:
            for m in ANSWER_INLINE_RE.finditer(line):
                q_num = int(m.group(1))
                answers[q_num] = m.group(2).upper()

    if answers:
        return answers

    # Strategy 3: Look for a dense block of answer entries at the very end
    last_lines = lines[-10:] if len(lines) >= 10 else lines
    candidate_answers = {}
    for line in last_lines:
        entries = ANSWER_INLINE_RE.findall(line)
        if len(entries) >= 3:
            for q_num_str, letter in entries:
                candidate_answers[int(q_num_str)] = letter.upper()

    # Only accept if we found a reasonable number
    if len(candidate_answers) >= 3:
        answers = candidate_answers

    return answers


def letter_to_option_index(letter: str) -> int:
    """Convert a letter (A, B, C, D) to option index (0, 1, 2, 3)."""
    return ord(letter.upper()) - ord('A')


def parse_options_from_text(text: str) -> List[str]:
    """
    Parse options from a block of text using multiple strategies.

    Tries different regex patterns in order of reliability.
    """
    options = []

    # Strategy 1: Bracket format [A] text [B] text
    matches = list(OPTION_BRACKET_RE.finditer(text))
    if len(matches) >= 2:
        for m in matches:
            opt = sanitize_option_text(m.group(2))
            if opt:
                options.append(opt)
        if len(options) >= 2:
            return options

    # Strategy 2: Lettered format A) text B) text or (A) text (B) text
    options = []
    matches = list(OPTION_LETTER_RE.finditer(text))
    if len(matches) >= 2:
        for m in matches:
            opt = sanitize_option_text(m.group(2))
            if opt:
                options.append(opt)
        if len(options) >= 2:
            return options

    # Strategy 3: Numbered format 1) text 2) text
    options = []
    matches = list(OPTION_NUMBER_RE.finditer(text))
    if len(matches) >= 2:
        for m in matches:
            opt = sanitize_option_text(m.group(2))
            if opt:
                options.append(opt)
        if len(options) >= 2:
            return options

    # Strategy 4: Simple split by option markers
    parts = re.split(r'(?:^|\n|\s)(?:\(?[A-Da-d]\)?|\d+)\s*[\)\.\]\:]\s*', text)
    options = [sanitize_option_text(p) for p in parts if p.strip()]
    if len(options) >= 2:
        return options

    return []


def parse_questions_from_text(text: str) -> List[CompletionQuestion]:
    """
    Parse sentence completion questions from extracted PDF text.

    Uses multiple strategies with fallback:
    1. Split by question numbers and parse each block
    2. Fallback: line-by-line parsing

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        List of validated CompletionQuestion objects.
    """
    answer_key = extract_answer_key(text)

    # Remove answer key section from text to avoid confusing the parser
    parse_text = text
    if answer_key:
        # Remove lines that look like answer key
        lines = text.split('\n')
        cutoff = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - 20, -1), -1):
            line = lines[i].strip()
            if re.match(r'^(answer|key|solutions?)', line, re.IGNORECASE):
                cutoff = i
                break
            entries = ANSWER_INLINE_RE.findall(line)
            if len(entries) >= 3:
                cutoff = i
                break
        parse_text = '\n'.join(lines[:cutoff])

    # Strategy 1: Split by question number markers
    questions = _parse_by_question_numbers(parse_text, answer_key)

    if len(questions) >= 2:
        return _validate_and_filter(questions)

    # Strategy 2: Fallback — line-by-line parsing
    logger.info("Primary parsing strategy yielded few results. Trying fallback.")
    questions = _parse_line_by_line(parse_text, answer_key)

    return _validate_and_filter(questions)


def _parse_by_question_numbers(text: str,
                                answer_key: Dict[int, str]) -> List[CompletionQuestion]:
    """Parse by splitting text at question number markers."""
    questions = []

    # Find all question number positions
    q_matches = list(Q_NUM_RE.finditer(text))
    if len(q_matches) < 1:
        return questions

    for i, q_match in enumerate(q_matches):
        q_num = int(q_match.group(1))
        start = q_match.end()
        end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)

        block = text[start:end].strip()

        # Find where options start
        opt_start = _find_options_start(block)
        if opt_start is None:
            continue

        question_text = block[:opt_start].strip()
        options_text = block[opt_start:].strip()

        # Normalize and validate
        question_text = sanitize_question_text(question_text)
        question_text = normalize_blank(question_text)

        if not has_blank(question_text):
            continue

        options = parse_options_from_text(options_text)
        if len(options) < 2:
            continue

        # Map answer key
        correct = None
        if q_num in answer_key:
            idx = letter_to_option_index(answer_key[q_num])
            if idx < len(options):
                correct = options[idx]

        q = CompletionQuestion(
            question=question_text,
            options=options,
            correct=correct,
        )
        questions.append(q)

    return questions


def _parse_line_by_line(text: str,
                         answer_key: Dict[int, str]) -> List[CompletionQuestion]:
    """Fallback: parse line by line, accumulating question and options."""
    questions = []
    lines = text.split('\n')

    current_q_num = None
    current_q_text = ""
    current_options = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this line starts a new question
        q_match = Q_NUM_RE.match('\n' + line)
        if q_match:
            # Save previous question if complete
            if current_q_text and current_options:
                q = _make_question(current_q_num, current_q_text,
                                   current_options, answer_key)
                if q:
                    questions.append(q)

            current_q_num = int(q_match.group(1))
            current_q_text = line[q_match.end() - len('\n'):].strip()
            current_q_text = sanitize_question_text(current_q_text)
            current_options = []
            continue

        # Check if this line is an option
        opt_match = re.match(r'^\(?([A-Da-d])\)?\s*[\)\.\]\:]\s*(.+)', line)
        if opt_match and current_q_text:
            opt = sanitize_option_text(opt_match.group(2))
            if opt:
                current_options.append(opt)
            continue

        opt_match_num = re.match(r'^(\d+)\s*[\)\.\]\:]\s*(.+)', line)
        if opt_match_num and current_q_text:
            opt = sanitize_option_text(opt_match_num.group(2))
            if opt:
                current_options.append(opt)
            continue

        opt_match_bracket = re.match(r'^\[([A-Da-d])\]\s*(.+)', line)
        if opt_match_bracket and current_q_text:
            opt = sanitize_option_text(opt_match_bracket.group(2))
            if opt:
                current_options.append(opt)
            continue

        # If we have a question but no options yet, this might be continuation
        if current_q_text and not current_options:
            current_q_text += " " + line
            current_q_text = sanitize_question_text(current_q_text)

    # Don't forget the last question
    if current_q_text and current_options:
        q = _make_question(current_q_num, current_q_text,
                           current_options, answer_key)
        if q:
            questions.append(q)

    return questions


def _find_options_start(block: str) -> Optional[int]:
    """Find where options start in a question block."""
    # Look for first option marker
    patterns = [
        re.compile(r'(?:^|\n|\s)\(?[A-Da-d]\)?\s*[\)\.\]\:]'),
        re.compile(r'(?:^|\n|\s)\[[A-Da-d]\]'),
        re.compile(r'(?:^|\n|\s)\d+\s*[\)\.\]\:]\s'),
    ]

    earliest = None
    for pattern in patterns:
        match = pattern.search(block)
        if match:
            pos = match.start()
            if earliest is None or pos < earliest:
                earliest = pos

    return earliest


def _make_question(q_num: Optional[int], q_text: str,
                   options: List[str],
                   answer_key: Dict[int, str]) -> Optional[CompletionQuestion]:
    """Create a validated CompletionQuestion."""
    q_text = normalize_blank(q_text)
    if not has_blank(q_text):
        return None

    correct = None
    if q_num and q_num in answer_key:
        idx = letter_to_option_index(answer_key[q_num])
        if idx < len(options):
            correct = options[idx]

    return CompletionQuestion(
        question=q_text,
        options=options,
        correct=correct,
    )


def _validate_and_filter(questions: List[CompletionQuestion]) -> List[CompletionQuestion]:
    """Validate parsed questions and filter out invalid ones."""
    from src.utils.validation import validate_completion_question

    valid = []
    for i, q in enumerate(questions):
        is_valid, err = validate_completion_question(q)
        if is_valid:
            valid.append(q)
        else:
            logger.warning(f"Q{i + 1} skipped: {err}")

    logger.info(f"Parsed {len(questions)} questions, {len(valid)} valid.")
    return valid


def parse_pdf(pdf_file) -> List[CompletionQuestion]:
    """
    Full PDF parsing pipeline: extract text and parse questions.

    Args:
        pdf_file: File path (str) or file-like object (BytesIO).

    Returns:
        List of validated CompletionQuestion objects.

    Raises:
        ValueError: If PDF can't be read or no questions found.
    """
    text = extract_text_from_pdf(pdf_file)
    questions = parse_questions_from_text(text)

    if not questions:
        raise ValueError(
            "No sentence completion questions could be parsed from the PDF. "
            "Ensure questions follow a standard format with numbered questions "
            "and lettered (A/B/C/D) or numbered (1/2/3) options.\n"
            "Supported formats:\n"
            "  1. Question text _____\n"
            "     A) option1  B) option2  C) option3\n"
            "  1. Question text _____\n"
            "     (A) option1  (B) option2\n"
            "  1. Question text _____\n"
            "     [A] option1  [B] option2"
        )

    return questions


def parse_pdf_with_metadata(pdf_file) -> Tuple[List[CompletionQuestion], str]:
    """
    Parse PDF and return both questions and the raw extracted text.

    Useful for debugging and showing the user what was extracted.

    Returns:
        (questions, raw_text)
    """
    text = extract_text_from_pdf(pdf_file)
    questions = parse_questions_from_text(text)
    return questions, text
