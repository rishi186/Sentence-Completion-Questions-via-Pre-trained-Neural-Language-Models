<div align="center">

# ESL Sentence Completion

### via Pre-trained Neural Language Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-ffd21e)](https://huggingface.co/transformers)
[![Tests](https://img.shields.io/badge/Tests-48%20passing-success)](https://github.com/rishi186/Sentence-Completion-Questions-via-Pre-trained-Neural-Language-Models)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p>
  <b>Automatically solve ESL sentence completion questions</b><br>
  using BART, BERT, RoBERTa, and GPT-2 — with PDF upload, batch evaluation, and result export.
</p>

---

[Quick Start](#-quick-start) · [Models](#-supported-models) · [Web App](#-web-interface) · [CLI](#-cli-interface) · [PDF Support](#-pdf-support) · [Testing](#-testing) · [Docker](#-docker-deployment)

</div>

---

## Overview

Sentence completion (SC) questions present a sentence with one or more blanks and several candidate options. This project uses **pre-trained neural language models** to predict the most appropriate option — aligned with the research paper:

> Liu, Q., Liu, T., Zhao, J., Fang, Q., Ding, W., Wu, Z., Xia, F., Tang, J., & Liu, Z. *"Solving ESL Sentence Completion Questions via Pre-trained Neural Language Models."* arXiv:2107.07122 (2021).

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Scoring** | BART (Seq2Seq), BERT/RoBERTa (MLM), GPT-2 (CLM) |
| **PDF Upload & Parsing** | Upload question PDFs → auto-parse → answer → export |
| **Batch Evaluation** | Accuracy, per-position accuracy, bias analysis |
| **Fine-Tuning** | Fine-tune models on custom ESL datasets |
| **Web Interface** | Interactive Streamlit app with visualizations |
| **CLI** | Full command-line interface for all operations |
| **Input Validation** | Comprehensive validation with clear error messages |
| **Retry Logic** | Automatic retries for transient failures |
| **Production Ready** | Docker, logging, testing, packaging |
| **Result Export** | PDF, JSON, CSV download options |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rishi186/Sentence-Completion-Questions-via-Pre-trained-Neural-Language-Models.git
cd Sentence-Completion-Questions-via-Pre-trained-Neural-Language-Models

# 2. Install
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Supported Models

| Model Type | Architecture | Default Model | Scoring Method |
|------------|-------------|---------------|----------------|
| **BART** | Seq2Seq | `facebook/bart-large` | Encoder-decoder log-likelihood |
| **BERT** | MLM | `bert-base-uncased` | Masked language modeling |
| **RoBERTa** | MLM | `roberta-base` | Masked language modeling |
| **GPT-2** | CLM | `gpt2` | Causal language modeling |

## Web Interface

Launch the Streamlit app with `streamlit run app.py`. The app provides four tabs:

| Tab | Description |
|-----|-------------|
| **Single Question** | Test individual questions with score visualization (bar chart + table) |
| **PDF Upload & Answer** | Upload a PDF → auto-parse questions → batch answer → download results |
| **Batch Evaluation** | Upload a JSON dataset → run full evaluation with metrics & bias analysis |
| **About** | Project info, model comparison table, and reference paper |

## CLI Interface

```bash
# Predict a single question
python main.py predict \
  --question "The cat sat on the _____." \
  --options mat table chair \
  --model bart

# Evaluate on a dataset
python main.py evaluate \
  --data data/sample_questions.json \
  --model bert \
  --output results/

# Answer questions from a PDF
python main.py pdf \
  --input questions.pdf \
  --model bart \
  --output results/

# Fine-tune a model
python main.py finetune \
  --train-data data/train.json \
  --eval-data data/eval.json \
  --model-name facebook/bart-base \
  --epochs 3

# Interactive mode
python main.py interactive --model roberta
```

## PDF Support

Upload PDFs with sentence completion questions. The parser supports **multiple formats**:

<details>
<summary><b>Format 1 — Lettered options (each on own line)</b></summary>

```
1. The cat sat on the _____.
   A) mat
   B) table
   C) chair
   D) roof
```
</details>

<details>
<summary><b>Format 2 — Bracket options</b></summary>

```
1. The cat sat on the _____.
   [A] mat
   [B] table
   [C] chair
```
</details>

<details>
<summary><b>Format 3 — Inline options</b></summary>

```
1. The cat sat on the _____. A) mat B) table C) chair D) roof
```
</details>

<details>
<summary><b>Format 4 — Parenthesis options</b></summary>

```
1. The cat sat on the _____.
   (A) mat  (B) table  (C) chair  (D) roof
```
</details>

<details>
<summary><b>Answer Key (optional)</b></summary>

```
Answer Key: 1-A, 2-B, 3-C, 4-D
```

Or block format:
```
Answer Key:
1.A
2.B
3.C
```
</details>

### Export Options

After answering, download results as:
- **PDF** — Formatted document with questions, options, scores, and correctness
- **JSON** — Structured data for programmatic use
- **CSV** — Spreadsheet-friendly format

## Project Structure

```
Sentence-Completion-Questions-via-Pre-trained-Neural-Language/
├── main.py                    # CLI entry point
├── app.py                     # Streamlit web app
├── config.py                  # Configuration dataclasses
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Packaging & tooling config
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose config
├── Makefile                   # Dev commands (make test, make app, etc.)
├── .env.example               # Environment variable template
├── data/
│   ├── sample_questions.json  # 20 sample ESL questions
│   └── sample_questions.pdf   # Sample PDF for testing
├── src/
│   ├── models/
│   │   ├── sentence_completion.py  # Core scorers (BART, MLM, CLM)
│   │   └── fine_tune.py            # Fine-tuning module
│   ├── data/
│   │   ├── prepare.py              # Data preparation & question dataclass
│   │   ├── dataset.py              # Dataset loading (JSON/CSV)
│   │   └── pdf_parser.py           # PDF question parsing
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, bias analysis, per-position
│   └── utils/
│       ├── helpers.py              # Device selection, seeding, I/O
│       ├── logger.py               # Structured logging
│       ├── validation.py           # Input validation & sanitization
│       └── pdf_export.py           # Results to PDF export
├── tests/
│   └── test_core.py           # 48 unit tests
└── generate_sample_pdf.py     # Sample PDF generator
```

## Dataset Format

**JSON:**
```json
[
  {
    "question": "The cat sat on the _____.",
    "options": ["mat", "table", "chair", "roof"],
    "answer": "mat"
  }
]
```

**CSV** (options separated by `;`):
```csv
question,options,answer
The cat sat on the _____.,mat;table;chair;roof,mat
```

## Testing

```bash
# Run all 48 tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Test coverage includes:
- Data preparation & question dataclass
- Dataset loading (JSON, list, filtering)
- Evaluation metrics (accuracy, per-position, bias)
- Input validation (questions, options, duplicates, limits)
- PDF parsing (blank normalization, answer key, multiple formats)
- Scorer factory & error handling
- Logger infrastructure

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t esl-completion .
docker run -p 8501:8501 esl-completion
```

The app will be available at `http://localhost:8501`.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Available Make commands
make install      # Install dependencies
make dev-install  # Install with dev tools
make test         # Run tests
make test-cov     # Run tests with coverage
make lint         # Lint code
make format       # Format code
make app          # Run Streamlit app
make pdf          # Generate sample PDF
make clean        # Clean build artifacts
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Overall Accuracy** | Percentage of correctly answered questions |
| **Per-Position Accuracy** | Accuracy by correct answer position (A/B/C/D) |
| **Bias Analysis** | Statistical detection of positional bias |
| **Detailed Results** | Per-question scores and predictions |

## License

MIT License — see [LICENSE](LICENSE) for details.

## References

1. Liu, Q. et al. *"Solving ESL Sentence Completion Questions via Pre-trained Neural Language Models."* arXiv:2107.07122 (2021).
2. Devlin, J. et al. *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* (2019).
3. Lewis, M. et al. *"BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension."* (2019).
4. Radford, A. et al. *"Language Models are Unsupervised Multitask Learners."* (GPT-2, 2019).

---

<div align="center">

**If this project helped you, consider giving it a star!**

</div>
