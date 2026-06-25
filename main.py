"""CLI interface for ESL Sentence Completion.

Usage:
    python main.py predict --question "The cat sat on the _____." --options mat table chair
    python main.py predict --model bert --question "..." --options ...
    python main.py evaluate --data data/sample_questions.json --model bart
    python main.py evaluate --data data/sample_questions.json --model bert --output results/
    python main.py finetune --train-data data/train.json --eval-data data/eval.json --model facebook/bart-base
    python main.py pdf --input questions.pdf --model bart --output results/
"""

import argparse
import json
import os
import sys

from src.models.sentence_completion import create_scorer, DEFAULT_MODELS, ScorerError
from src.data.dataset import CompletionDataset
from src.data.prepare import CompletionQuestion
from src.data.pdf_parser import parse_pdf, parse_pdf_with_metadata
from src.evaluation.metrics import evaluate_dataset, bias_analysis
from src.models.fine_tune import FineTuner
from src.utils.helpers import set_seed, save_json, format_results
from src.utils.pdf_export import export_results_to_pdf
from src.utils.logger import get_logger
from src.utils.validation import validate_question, validate_options
from config import TrainConfig

logger = get_logger(__name__)


def _load_scorer_or_exit(args):
    """Load scorer with error handling. Exits on failure."""
    try:
        scorer = create_scorer(
            model_type=args.model,
            model_name=args.model_name,
            device=args.device,
        )
        return scorer
    except ScorerError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_predict(args):
    """Predict the best option for a single question."""
    set_seed(args.seed)

    # Validate inputs
    q_valid, q_err = validate_question(args.question)
    if not q_valid:
        print(f"Error: Invalid question: {q_err}", file=sys.stderr)
        sys.exit(1)

    o_valid, o_err = validate_options(args.options)
    if not o_valid:
        print(f"Error: Invalid options: {o_err}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model} ({args.model_name or 'default'})...")
    scorer = _load_scorer_or_exit(args)

    print(f"\nQuestion: {args.question}")
    print(f"Options: {args.options}")

    try:
        best_option, scores = scorer.predict(args.question, args.options)
    except (ScorerError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nScores:")
    for opt, score in zip(args.options, scores):
        marker = " <-- BEST" if opt == best_option else ""
        print(f"  {opt:20s}: {score:.6f}{marker}")
    print(f"\nBest Option: {best_option}")


def cmd_evaluate(args):
    """Evaluate model on a dataset."""
    set_seed(args.seed)
    print(f"Loading dataset from {args.data}...")
    try:
        dataset = CompletionDataset.from_json(args.data)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    dataset = dataset.filter_answered()
    if len(dataset) == 0:
        print("Error: No questions with known answers found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset)} questions with known answers.")

    print(f"\nLoading model: {args.model}...")
    scorer = _load_scorer_or_exit(args)

    print("\nEvaluating...")
    try:
        results = evaluate_dataset(scorer, list(dataset), verbose=True)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    bias = bias_analysis(results)
    if bias.get("has_bias"):
        print(f"\n⚠ Bias detected: {bias}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        save_json(results, os.path.join(args.output, "evaluation_results.json"))
        save_json(bias, os.path.join(args.output, "bias_analysis.json"))
        print(f"\nResults saved to {args.output}/")


def cmd_finetune(args):
    """Fine-tune a model on a dataset."""
    set_seed(args.seed)
    print(f"Loading training data from {args.train_data}...")
    train_dataset = CompletionDataset.from_json(args.train_data)
    train_questions = [q for q in train_dataset if q.correct is not None]
    print(f"  Training questions: {len(train_questions)}")

    eval_questions = None
    if args.eval_data:
        print(f"Loading eval data from {args.eval_data}...")
        eval_dataset = CompletionDataset.from_json(args.eval_data)
        eval_questions = [q for q in eval_dataset if q.correct is not None]
        print(f"  Eval questions: {len(eval_questions)}")

    config = TrainConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print(f"\nFine-tuning {args.model_name}...")
    finetuner = FineTuner(
        model_name=args.model_name,
        config=config,
        device=args.device,
    )
    history = finetuner.train(train_questions, eval_questions)

    print(f"\nTraining complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["eval_accuracy"]:
        print(f"  Final eval accuracy: {history['eval_accuracy'][-1]:.4f}")
    print(f"  Checkpoints saved to {args.output_dir}/")


def cmd_interactive(args):
    """Interactive mode for testing questions."""
    set_seed(args.seed)
    print(f"Loading model: {args.model}...")
    scorer = _load_scorer_or_exit(args)
    print("\nInteractive mode. Type 'quit' to exit.\n")

    while True:
        question = input("Question (use _____ for blank): ").strip()
        if question.lower() == "quit":
            break
        options_str = input("Options (comma-separated): ").strip()
        if options_str.lower() == "quit":
            break
        options = [o.strip() for o in options_str.split(",")]

        # Validate
        q_valid, q_err = validate_question(question)
        if not q_valid:
            print(f"  Invalid question: {q_err}")
            continue
        o_valid, o_err = validate_options(options)
        if not o_valid:
            print(f"  Invalid options: {o_err}")
            continue

        try:
            best, scores = scorer.predict(question, options)
        except (ScorerError, ValueError) as e:
            print(f"  Error: {e}")
            continue

        print(f"\nScores:")
        for opt, score in zip(options, scores):
            marker = " <-- BEST" if opt == best else ""
            print(f"  {opt:20s}: {score:.6f}{marker}")
        print(f"Best Option: {best}\n")


def cmd_pdf(args):
    """Parse questions from a PDF, answer them, and export results."""
    set_seed(args.seed)
    print(f"Parsing PDF: {args.input}...")
    try:
        questions, raw_text = parse_pdf_with_metadata(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error parsing PDF: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(questions)} questions.")

    if not questions:
        print("No questions found in the PDF.")
        return

    # Print parsed questions
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q.question}")
        for j, opt in enumerate(q.options):
            letter = chr(65 + j)
            print(f"    {letter}) {opt}")
        if q.correct:
            print(f"    Answer: {q.correct}")

    print(f"\nLoading model: {args.model}...")
    scorer = _load_scorer_or_exit(args)

    print("\nAnswering questions...")
    all_results = []
    for i, q in enumerate(questions):
        try:
            best_option, scores = scorer.predict(q.question, q.options, q.blank_token)
        except (ScorerError, ValueError) as e:
            logger.error(f"Q{i + 1} failed: {e}")
            best_option = q.options[0] if q.options else ""
            scores = [float("-inf")] * len(q.options)

        result = {
            "question": q.question,
            "options": q.options,
            "scores": [round(s, 6) for s in scores],
            "predicted": best_option,
        }
        if q.correct:
            result["correct"] = q.correct
            result["is_correct"] = best_option == q.correct
        all_results.append(result)

        correct_marker = ""
        if q.correct:
            correct_marker = " ✓" if best_option == q.correct else f" ✗ (correct: {q.correct})"
        print(f"  Q{i+1}: {best_option}{correct_marker}")

    # Summary
    has_answers = any(r.get("correct") for r in all_results)
    if has_answers:
        correct_count = sum(1 for r in all_results if r.get("is_correct"))
        print(f"\nAccuracy: {correct_count}/{len(all_results)} ({correct_count/len(all_results):.1%})")

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        json_path = os.path.join(args.output, "answered_questions.json")
        pdf_path = os.path.join(args.output, "answered_questions.pdf")
        save_json(all_results, json_path)
        try:
            export_results_to_pdf(all_results, pdf_path)
            print(f"\nResults saved to {args.output}/")
            print(f"  JSON: {json_path}")
            print(f"  PDF:  {pdf_path}")
        except Exception as e:
            print(f"  PDF export failed: {e}", file=sys.stderr)
            print(f"  JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ESL Sentence Completion via Pre-trained Neural Language Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict best option for a question")
    pred_parser.add_argument("--question", required=True, help="Question with _____ blank")
    pred_parser.add_argument("--options", nargs="+", required=True, help="List of options")
    pred_parser.add_argument("--model", default="bart", choices=["bart", "bert", "roberta", "gpt2"])
    pred_parser.add_argument("--model-name", default=None, help="Specific HuggingFace model name")
    pred_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pred_parser.add_argument("--seed", type=int, default=42)
    pred_parser.set_defaults(func=cmd_predict)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    eval_parser.add_argument("--data", required=True, help="Path to JSON dataset")
    eval_parser.add_argument("--model", default="bart", choices=["bart", "bert", "roberta", "gpt2"])
    eval_parser.add_argument("--model-name", default=None, help="Specific HuggingFace model name")
    eval_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    eval_parser.add_argument("--output", default=None, help="Output directory for results")
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.set_defaults(func=cmd_evaluate)

    # Fine-tune command
    ft_parser = subparsers.add_parser("finetune", help="Fine-tune a model")
    ft_parser.add_argument("--train-data", required=True, help="Path to training JSON")
    ft_parser.add_argument("--eval-data", default=None, help="Path to eval JSON")
    ft_parser.add_argument("--model-name", default="facebook/bart-base", help="Base model to fine-tune")
    ft_parser.add_argument("--epochs", type=int, default=3)
    ft_parser.add_argument("--lr", type=float, default=2e-5)
    ft_parser.add_argument("--batch-size", type=int, default=8)
    ft_parser.add_argument("--output-dir", default="checkpoints")
    ft_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ft_parser.add_argument("--seed", type=int, default=42)
    ft_parser.set_defaults(func=cmd_finetune)

    # PDF command
    pdf_parser = subparsers.add_parser("pdf", help="Parse PDF, answer questions, and export results")
    pdf_parser.add_argument("--input", required=True, help="Path to PDF file")
    pdf_parser.add_argument("--model", default="bart", choices=["bart", "bert", "roberta", "gpt2"])
    pdf_parser.add_argument("--model-name", default=None, help="Specific HuggingFace model name")
    pdf_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pdf_parser.add_argument("--output", default=None, help="Output directory for results")
    pdf_parser.add_argument("--seed", type=int, default=42)
    pdf_parser.set_defaults(func=cmd_pdf)

    # Interactive command
    int_parser = subparsers.add_parser("interactive", help="Interactive prediction mode")
    int_parser.add_argument("--model", default="bart", choices=["bart", "bert", "roberta", "gpt2"])
    int_parser.add_argument("--model-name", default=None, help="Specific HuggingFace model name")
    int_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    int_parser.add_argument("--seed", type=int, default=42)
    int_parser.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
