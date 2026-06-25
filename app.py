"""Streamlit web application for ESL Sentence Completion.

Run with: streamlit run app.py

Production features:
- Session state management for model caching
- Graceful error handling with user-friendly messages
- Progress indicators for long operations
- PDF upload, parsing, answering, and export
- Batch evaluation with metrics and bias analysis
- Model status indicators
"""

import streamlit as st
import pandas as pd
import json
import os
import io
import time

from src.models.sentence_completion import (
    create_scorer, DEFAULT_MODELS, ScorerError,
)
from src.data.dataset import CompletionDataset
from src.data.prepare import CompletionQuestion
from src.data.pdf_parser import parse_pdf_with_metadata
from src.evaluation.metrics import evaluate_dataset, bias_analysis
from src.utils.helpers import set_seed
from src.utils.pdf_export import export_results_to_pdf
from src.utils.logger import get_logger
from src.utils.validation import validate_question, validate_options

logger = get_logger(__name__)


# --- Session state initialization ---

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "scorer": None,
        "scorer_model_type": None,
        "scorer_model_name": None,
        "pdf_questions": None,
        "pdf_raw_text": None,
        "pdf_results": None,
        "eval_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_scorer(model_type: str, model_name: str = None):
    """Get or create a cached scorer in session state."""
    if (st.session_state.scorer is not None
            and st.session_state.scorer_model_type == model_type
            and st.session_state.scorer_model_name == model_name):
        return st.session_state.scorer

    set_seed(42)
    try:
        scorer = create_scorer(
            model_type=model_type,
            model_name=model_name,
            device="auto",
        )
        st.session_state.scorer = scorer
        st.session_state.scorer_model_type = model_type
        st.session_state.scorer_model_name = model_name
        return scorer
    except ScorerError as e:
        st.error(f"Failed to load model: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        logger.error(f"Model load error: {e}", exc_info=True)
        return None


def main():
    st.set_page_config(
        page_title="ESL Sentence Completion",
        page_icon="📝",
        layout="wide",
    )

    init_session_state()

    st.title("ESL Sentence Completion via Pre-trained Neural Language Models")
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        model_type = st.selectbox(
            "Model Type",
            options=["bart", "bert", "roberta", "gpt2"],
            help="BART: Seq2Seq | BERT/RoBERTa: MLM | GPT-2: CLM",
        )
        model_name = st.text_input(
            "Model Name (optional)",
            value="",
            help=f"Leave empty for default: {DEFAULT_MODELS.get(model_type, 'N/A')}",
        ).strip() or None

        # Model status indicator
        if st.session_state.scorer is not None:
            if st.session_state.scorer.is_loaded:
                st.success(f"✅ Model loaded: {st.session_state.scorer_model_type.upper()}")
            else:
                st.warning("⚠ Model not loaded")
        else:
            st.info("⚪ No model loaded yet")

        st.markdown("---")
        st.caption("ESL Sentence Completion v1.0")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Question", "PDF Upload & Answer", "Batch Evaluation", "About",
    ])

    # === Tab 1: Single Question ===
    with tab1:
        st.header("Test a Single Question")

        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Question (use _____ for the blank)",
                value="The cat sat on the _____.",
                key="single_question",
            )
        with col2:
            st.info("Blank token: `_____`")

        options_text = st.text_input(
            "Options (comma-separated)",
            value="mat, table, chair",
            key="single_options",
        )
        options = [o.strip() for o in options_text.split(",") if o.strip()]

        if st.button("Predict", type="primary", key="btn_predict"):
            # Validate inputs
            q_valid, q_err = validate_question(question)
            if not q_valid:
                st.error(f"Invalid question: {q_err}")
                return

            o_valid, o_err = validate_options(options)
            if not o_valid:
                st.error(f"Invalid options: {o_err}")
                return

            # Load model
            with st.spinner(f"Loading {model_type.upper()} model..."):
                scorer = get_scorer(model_type, model_name)

            if scorer is None:
                return

            # Score
            with st.spinner("Scoring options..."):
                try:
                    best_option, scores = scorer.predict(question, options)
                except ScorerError as e:
                    st.error(f"Scoring failed: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    return

            # Display results
            st.success(f"Best Option: **{best_option}**")

            results_df = pd.DataFrame({
                "Option": options,
                "Score": scores,
            })
            results_df = results_df.sort_values("Score", ascending=False)
            results_df["Rank"] = range(1, len(results_df) + 1)

            col_chart, col_table = st.columns([1, 1])
            with col_table:
                st.subheader("Scores")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            with col_chart:
                st.subheader("Score Comparison")
                st.bar_chart(results_df.set_index("Option")["Score"])

    # === Tab 2: PDF Upload & Answer ===
    with tab2:
        st.header("Upload PDF & Get Answers")
        st.markdown(
            "Upload a PDF with sentence completion questions. "
            "The app will parse them, answer each one, and export results."
        )

        with st.expander("Supported PDF format", expanded=False):
            st.code("""1. The cat sat on the _____.
   A) mat
   B) table
   C) chair
   D) roof

2. An apple is a _____.
   A) fruit
   B) vegetable
   C) fish

Answer Key: 1-A, 2-A  (optional)""", language="text")

        pdf_file = st.file_uploader(
            "Upload PDF file",
            type=["pdf"],
            help="PDF with numbered questions and lettered (A/B/C/D) options",
            key="pdf_upload",
        )

        if pdf_file is not None:
            # Parse PDF
            try:
                with st.spinner("Parsing PDF..."):
                    pdf_bytes = io.BytesIO(pdf_file.read())
                    questions, raw_text = parse_pdf_with_metadata(pdf_bytes)

                st.session_state.pdf_questions = questions
                st.session_state.pdf_raw_text = raw_text
                st.session_state.pdf_results = None

                st.success(f"Successfully parsed {len(questions)} questions from PDF!")

                # Preview parsed questions
                with st.expander(f"Preview parsed questions ({len(questions)} found)",
                                 expanded=True):
                    for i, q in enumerate(questions, 1):
                        st.write(f"**Q{i}:** {q.question}")
                        for j, opt in enumerate(q.options):
                            letter = chr(65 + j)
                            marker = " ✓" if q.correct == opt else ""
                            st.write(f"  {letter}) {opt}{marker}")
                        if q.correct:
                            st.write(f"  *Answer: {q.correct}*")
                        st.write("")

                # Debug: raw text
                with st.expander("Raw extracted text (for debugging)", expanded=False):
                    st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))

            except ValueError as e:
                st.error(f"PDF parsing error: {e}")
                st.info("Make sure your PDF has text (not scanned images) "
                        "and follows the expected format.")
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.error(f"PDF parse error: {e}", exc_info=True)
                return

            # Answer questions
            if st.button("Answer All Questions", type="primary", key="btn_answer_pdf"):
                with st.spinner(f"Loading {model_type.upper()} model..."):
                    scorer = get_scorer(model_type, model_name)

                if scorer is None:
                    return

                all_results = []
                progress = st.progress(0.0, "Answering questions...")
                status_text = st.empty()

                for i, q in enumerate(questions):
                    try:
                        best_option, scores = scorer.predict(
                            q.question, q.options, q.blank_token
                        )
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

                    progress.progress(
                        (i + 1) / len(questions),
                        f"Answered {i + 1}/{len(questions)}",
                    )

                progress.empty()
                status_text.success(f"All {len(questions)} questions answered!")
                st.session_state.pdf_results = all_results

            # Display results if available
            if st.session_state.pdf_results:
                all_results = st.session_state.pdf_results

                # Summary metrics
                has_answers = any(r.get("correct") for r in all_results)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Questions", len(all_results))
                with col2:
                    if has_answers:
                        correct_count = sum(1 for r in all_results if r.get("is_correct"))
                        st.metric("Correct", f"{correct_count}/{len(all_results)}")
                    else:
                        st.metric("Answered", len(all_results))
                with col3:
                    if has_answers:
                        acc = correct_count / len(all_results)
                        st.metric("Accuracy", f"{acc:.1%}")
                    else:
                        st.metric("Model", model_type.upper())

                # Results table
                st.subheader("Results")
                display_rows = []
                for i, r in enumerate(all_results, 1):
                    row = {
                        "#": i,
                        "Question": r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
                        "Predicted": r["predicted"],
                    }
                    if r.get("correct"):
                        row["Correct"] = r["correct"]
                        row["✓"] = "✓" if r.get("is_correct") else "✗"
                    display_rows.append(row)

                st.dataframe(pd.DataFrame(display_rows), use_container_width=True,
                             hide_index=True)

                # Detailed view
                st.subheader("Detailed Answers")
                for i, r in enumerate(all_results, 1):
                    pred_short = r['predicted'][:30]
                    with st.expander(f"Q{i}: {r['question'][:70]}... → {pred_short}"):
                        st.write(f"**Question:** {r['question']}")
                        for j, (opt, score) in enumerate(zip(r["options"], r["scores"])):
                            letter = chr(65 + j)
                            is_pred = opt == r["predicted"]
                            is_corr = r.get("correct") == opt
                            if is_pred and is_corr:
                                st.write(f"  **{letter}) {opt}** — score: {score:.4f} ✓ (predicted, correct)")
                            elif is_pred:
                                st.write(f"  **{letter}) {opt}** — score: {score:.4f} ← predicted")
                            elif is_corr:
                                st.write(f"  {letter}) {opt} — score: {score:.4f} ✓ correct answer")
                            else:
                                st.write(f"  {letter}) {opt} — score: {score:.4f}")

                # Download buttons
                st.subheader("Download Results")
                col_dl1, col_dl2, col_dl3 = st.columns(3)

                with col_dl1:
                    try:
                        pdf_bytes_out = export_results_to_pdf(all_results)
                        st.download_button(
                            "📄 Download PDF",
                            data=pdf_bytes_out,
                            file_name="answered_questions.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"PDF export failed: {e}")

                with col_dl2:
                    json_out = json.dumps(all_results, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📋 Download JSON",
                        data=json_out,
                        file_name="answered_questions.json",
                        mime="application/json",
                    )

                with col_dl3:
                    csv_data = pd.DataFrame([
                        {"#": i + 1, "question": r["question"],
                         "predicted": r["predicted"],
                         "correct": r.get("correct", ""),
                         "is_correct": r.get("is_correct", "")}
                        for i, r in enumerate(all_results)
                    ]).to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        data=csv_data,
                        file_name="answered_questions.csv",
                        mime="text/csv",
                    )

    # === Tab 3: Batch Evaluation ===
    with tab3:
        st.header("Evaluate on a Dataset")

        uploaded_file = st.file_uploader(
            "Upload JSON dataset",
            type=["json"],
            help='Format: [{"question": "...", "options": [...], "answer": "..."}]',
            key="batch_upload",
        )

        use_sample = st.checkbox("Use sample dataset instead", key="use_sample")

        if st.button("Run Evaluation", type="primary", key="btn_eval"):
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    dataset = CompletionDataset.from_list(data)
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                    return
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
                    return
            elif use_sample:
                sample_path = "data/sample_questions.json"
                if os.path.exists(sample_path):
                    try:
                        dataset = CompletionDataset.from_json(sample_path)
                    except Exception as e:
                        st.error(f"Failed to load sample: {e}")
                        return
                else:
                    st.error("Sample dataset not found.")
                    return
            else:
                st.warning("Please upload a dataset or use the sample.")
                return

            dataset = dataset.filter_answered()
            if len(dataset) == 0:
                st.warning("No questions with known answers found.")
                return

            st.info(f"Evaluating on {len(dataset)} questions...")

            with st.spinner(f"Loading {model_type.upper()} model..."):
                scorer = get_scorer(model_type, model_name)

            if scorer is None:
                return

            with st.spinner("Evaluating..."):
                try:
                    results = evaluate_dataset(scorer, list(dataset), verbose=False)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logger.error(f"Eval error: {e}", exc_info=True)
                    return

            st.session_state.eval_results = results

        # Display eval results if available
        if st.session_state.eval_results:
            results = st.session_state.eval_results

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", results["total_questions"])
            with col2:
                st.metric("Accuracy", f"{results['accuracy']:.2%}")
            with col3:
                bias = bias_analysis(results)
                st.metric("Bias Detected", "Yes" if bias.get("has_bias") else "No")

            st.subheader("Per-Position Accuracy")
            pos_df = pd.DataFrame(
                [(f"Position {k}", v) for k, v in results["per_position_accuracy"].items()],
                columns=["Position", "Accuracy"],
            )
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

            st.subheader("Detailed Results")
            detail_df = pd.DataFrame(results["detailed_results"])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name="evaluation_results.json",
                mime="application/json",
                key="dl_eval",
            )

    # === Tab 4: About ===
    with tab4:
        st.header("About This Project")
        st.markdown("""
        This project implements a solution for **ESL (English as a Second Language)
        sentence completion questions** using pre-trained neural language models.

        ### Supported Models

        | Model Type | Architecture | Scoring Method |
        |------------|-------------|---------------|
        | BART | Seq2Seq | Encoder-decoder log-likelihood |
        | BERT | MLM | Masked language modeling |
        | RoBERTa | MLM | Masked language modeling |
        | GPT-2 | CLM | Causal language modeling |

        ### Features

        - **Multiple model architectures** aligned with the research paper
        - **PDF upload** with automatic question parsing and answering
        - **Batch evaluation** with accuracy and bias analysis
        - **Fine-tuning support** via CLI
        - **Result export** to PDF, JSON, and CSV
        - **Input validation** and error handling
        - **Retry logic** for transient failures

        ### Reference

        Liu, Q. et al. "Solving ESL Sentence Completion Questions via Pre-trained
        Neural Language Models." arXiv:2107.07122 (2021).
        """)


if __name__ == "__main__":
    main()
