"""Export answered questions to PDF and other formats."""

import io
import json
from typing import List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak,
)
from reportlab.lib.enums import TA_LEFT


def export_results_to_pdf(results: List[Dict[str, Any]], output_path: str = None) -> bytes:
    """
    Export question answering results to a PDF file.

    Args:
        results: List of result dicts with keys: question, options, scores,
                 predicted, correct (optional), is_correct (optional).
        output_path: If provided, save to this path. Otherwise return bytes.

    Returns:
        PDF bytes if output_path is None, else None.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
        textColor=HexColor('#1a56db'),
    )
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold',
    )
    option_style = ParagraphStyle(
        'Option',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=2,
        leftIndent=20,
    )
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        spaceBefore=4,
        fontName='Helvetica-Bold',
        textColor=HexColor('#059669'),
    )
    correct_style = ParagraphStyle(
        'CorrectIndicator',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=2,
        leftIndent=20,
        textColor=HexColor('#059669'),
    )
    wrong_style = ParagraphStyle(
        'WrongIndicator',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=2,
        leftIndent=20,
        textColor=HexColor('#dc2626'),
    )
    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        fontName='Helvetica-Bold',
    )

    elements = []

    # Title
    elements.append(Paragraph("ESL Sentence Completion — Results", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Summary
    total = len(results)
    answered = sum(1 for r in results if r.get('predicted'))
    correct_count = sum(1 for r in results if r.get('is_correct') is True)
    has_answers = any(r.get('correct') for r in results)

    elements.append(Paragraph(f"Total Questions: {total}", summary_style))
    if has_answers:
        accuracy = correct_count / total if total > 0 else 0
        elements.append(Paragraph(f"Correct Answers: {correct_count}/{total} ({accuracy:.1%})", summary_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Each question
    for i, result in enumerate(results, 1):
        # Question
        q_text = result.get('question', '')
        q_text_escaped = q_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(f"Q{i}. {q_text_escaped}", question_style))

        # Options with scores
        options = result.get('options', [])
        scores = result.get('scores', [])
        predicted = result.get('predicted', '')
        correct = result.get('correct')

        for j, opt in enumerate(options):
            opt_escaped = opt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            letter = chr(65 + j)  # A, B, C, D

            score_str = f" (score: {scores[j]:.4f})" if j < len(scores) else ""
            is_predicted = (opt == predicted)
            is_correct_opt = (correct and opt == correct)

            if is_predicted and is_correct_opt:
                marker = " ✓ (predicted, correct)"
                style = correct_style
            elif is_predicted and correct and not is_correct_opt:
                marker = " ✗ (predicted, wrong)"
                style = wrong_style
            elif is_predicted:
                marker = " ← predicted"
                style = answer_style
            elif is_correct_opt:
                marker = " ✓ (correct answer)"
                style = correct_style
            else:
                marker = ""
                style = option_style

            elements.append(Paragraph(f"{letter}) {opt_escaped}{score_str}{marker}", style))

        # Final answer line
        if predicted:
            pred_escaped = predicted.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if correct and predicted != correct:
                correct_escaped = correct.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(
                    f"Predicted: {pred_escaped} | Correct: {correct_escaped}",
                    wrong_style,
                ))
            else:
                elements.append(Paragraph(f"Answer: {pred_escaped}", answer_style))

        elements.append(Spacer(1, 0.15 * inch))

    doc.build(elements)

    if output_path is None:
        buffer.seek(0)
        return buffer.getvalue()
    return None
