# ESL Sentence Completion via Pre-trained Neural Language Models

This project implements a solution for automatically solving English as a Second Language (ESL) sentence completion questions using pre-trained neural language models, based on the research paper:

> Liu, Q., Liu, T., Zhao, J., Fang, Q., Ding, W., Wu, Z., Xia, F., Tang, J., & Liu, Z. "Solving ESL Sentence Completion Questions via Pre-trained Neural Language Models." arXiv:2107.07122 (2021).

## Overview

Sentence completion (SC) questions present a sentence with one or more blanks that need to be filled in, with several possible options provided. This project uses pre-trained language models to predict the most appropriate option to complete the sentence.

## Features

- **Multiple Model Architectures**: Supports BART (Seq2Seq), BERT/RoBERTa (MLM), and GPT-2 (CLM) scoring approaches
- **Batch Evaluation**: Evaluate models on datasets with accuracy and bias analysis
- **Fine-tuning Support**: Fine-tune models on custom ESL datasets
- **Web Interface**: Interactive Streamlit app for testing and evaluation
- **CLI Interface**: Full command-line interface for all operations
- **Dataset Support**: Load from JSON or CSV, with sample dataset included

## Project Structure

```
Sentence-Completion-Questions-via-Pre-trained-Neural-Language/
├── main.py                          # CLI entry point
├── app.py                           # Streamlit web app
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
├── data/
│   └── sample_questions.json        # 20 sample ESL questions
├── src/
│   ├── models/
│   │   ├── sentence_completion.py   # Core model scorers (BART, MLM, CLM)
│   │   └── fine_tune.py             # Fine-tuning module
│   ├── data/
│   │   ├── prepare.py               # Data preparation utilities
│   │   └── dataset.py               # Dataset loading and management
│   ├── evaluation/
│   │   └── metrics.py               # Evaluation metrics and bias analysis
│   └── utils/
│       └── helpers.py               # Utility functions
└── tests/
    └── test_core.py                 # Unit tests
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishi186/Sentence-Completion-Questions-via-Pre-trained-Neural-Language-Models.git
   cd Sentence-Completion-Questions-via-Pre-trained-Neural-Language-Models
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### CLI Interface

**Predict a single question:**
```bash
python main.py predict --question "The cat sat on the _____." --options mat table chair --model bart
```

**Evaluate on a dataset:**
```bash
python main.py evaluate --data data/sample_questions.json --model bert --output results/
```

**Fine-tune a model:**
```bash
python main.py finetune --train-data data/train.json --eval-data data/eval.json --model-name facebook/bart-base --epochs 3
```

**Interactive mode:**
```bash
python main.py interactive --model roberta
```

### Web Interface

Launch the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to the displayed URL. The web app provides:
- **Single Question**: Test individual questions with score visualization
- **Batch Evaluation**: Upload a dataset and run full evaluation with metrics
- **About**: Project information and model comparison

### Supported Models

| Model Type | Default Model | Scoring Method |
|------------|--------------|---------------|
| `bart` | `facebook/bart-large` | Seq2Seq encoder-decoder log-likelihood |
| `bert` | `bert-base-uncased` | Masked language modeling (MLM) |
| `roberta` | `roberta-base` | Masked language modeling (MLM) |
| `gpt2` | `gpt2` | Causal language modeling (CLM) |

### Dataset Format

JSON format:
```json
[
  {
    "question": "The cat sat on the _____.",
    "options": ["mat", "table", "chair", "roof"],
    "answer": "mat"
  }
]
```

CSV format (columns: `question`, `options`, `answer`):
```csv
question,options,answer
The cat sat on the _____.,mat;table;chair,mat
```

## Scoring Approaches

### BART (Seq2Seq)
The encoder receives the sentence with the blank, and the decoder scores each option by computing the log-likelihood of the option tokens. Scores are normalized by token count to avoid length bias.

### BERT/RoBERTa (MLM)
The blank is replaced with the mask token. For single-token options, the probability is read directly from the mask position. For multi-token options, a pseudo-log-likelihood approach is used: each option token is masked in the full sentence and its probability is computed.

### GPT-2 (CLM)
The log-likelihood of the option tokens is computed given the preceding context. This leverages the autoregressive nature of causal language models.

## Evaluation

The evaluation module provides:
- **Overall accuracy**: Percentage of correctly answered questions
- **Per-position accuracy**: Accuracy broken down by the position of the correct answer (to detect positional bias)
- **Confusion matrix**: Which options get confused with each other
- **Bias analysis**: Statistical analysis of positional bias

## Testing

Run the unit tests:
```bash
python -m pytest tests/ -v
```

Or with unittest:
```bash
python -m unittest tests.test_core -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

1. Liu, Q. et al. "Solving ESL Sentence Completion Questions via Pre-trained Neural Language Models." arXiv:2107.07122 (2021).
