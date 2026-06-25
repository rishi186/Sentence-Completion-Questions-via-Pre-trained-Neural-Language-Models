"""Generate a sample PDF with sentence completion questions for testing."""

import os
from reportlab.lib.pagesizes import letter as page_letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def generate_sample_pdf(output_path: str = "data/sample_questions.pdf") -> str:
    """Generate a sample PDF with ESL sentence completion questions."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=page_letter,
                           rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                           topMargin=0.75 * inch, bottomMargin=0.75 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=20)
    q_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=11,
                             spaceBefore=12, spaceAfter=4, fontName='Helvetica-Bold')
    opt_style = ParagraphStyle('Option', parent=styles['Normal'], fontSize=10,
                               spaceAfter=2, leftIndent=24)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=10,
                                  spaceBefore=20, fontName='Helvetica-Bold')

    questions = [
        {
            "question": "The cat sat on the _____.",
            "options": ["mat", "table", "chair", "roof"],
            "answer": "A",
        },
        {
            "question": "An apple is a _____.",
            "options": ["fruit", "vegetable", "fish", "car"],
            "answer": "A",
        },
        {
            "question": "The teacher asked the students to _____ their homework.",
            "options": ["submit", "eat", "drive", "paint"],
            "answer": "A",
        },
        {
            "question": "Water _____ at 100 degrees Celsius.",
            "options": ["boils", "freezes", "flies", "sings"],
            "answer": "A",
        },
        {
            "question": "She was very _____ about passing the exam.",
            "options": ["happy", "wooden", "triangular", "metallic"],
            "answer": "A",
        },
        {
            "question": "The sun _____ in the east.",
            "options": ["rises", "sets", "falls", "sinks"],
            "answer": "A",
        },
        {
            "question": "He _____ a book every night before bed.",
            "options": ["reads", "throws", "eats", "builds"],
            "answer": "A",
        },
        {
            "question": "The restaurant is famous for its delicious _____.",
            "options": ["food", "cars", "furniture", "weather"],
            "answer": "A",
        },
        {
            "question": "Plants need _____ to grow.",
            "options": ["water", "plastic", "metal", "electricity"],
            "answer": "A",
        },
        {
            "question": "The baby _____ when she was hungry.",
            "options": ["cried", "drove", "typed", "swam"],
            "answer": "A",
        },
    ]

    elements = []
    elements.append(Paragraph("ESL Sentence Completion Test", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    for i, q in enumerate(questions, 1):
        elements.append(Paragraph(f"{i}. {q['question']}", q_style))
        for j, opt in enumerate(q["options"]):
            letter = chr(65 + j)
            elements.append(Paragraph(f"{letter}) {opt}", opt_style))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Answer Key", answer_style))
    answer_line = ", ".join(f"{i}-{q['answer']}" for i, q in enumerate(questions, 1))
    elements.append(Paragraph(answer_line, styles['Normal']))

    doc.build(elements)
    print(f"Sample PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()
