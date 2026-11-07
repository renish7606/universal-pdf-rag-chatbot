"""Reproducible Phase-0 benchmark for the local PDF RAG pipeline.

Usage: python evaluation/run_benchmark.py --pdf path/to/ml_ebook.pdf
Results are cached per PDF/configuration in evaluation/results/.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag
from langchain_core.prompts import ChatPromptTemplate

QUESTIONS_PATH = Path(__file__).with_name("benchmark_questions.json")
RESULTS_DIR = Path(__file__).with_name("results")
REFUSAL = "This information is not covered in the provided PDF."
JUDGE_PROMPT = ChatPromptTemplate.from_template("""Judge this PDF-RAG answer. Use the reference and retrieved context only.
Question: {question}
Reference: {reference}
Context: {context}
Answer: {answer}
Return JSON only: {{\"score\": 0|1|2, \"reason\": \"brief reason\"}}.
Score 2 only when correct and grounded, 1 for partial, 0 for incorrect or unsupported. For an out-of-document question, only the exact refusal is correct.""")


def load_questions():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if len(questions) != 60 or sum(not q["expected_pages"] for q in questions) != 10:
        raise ValueError("benchmark_questions.json must contain 60 questions: 50 answerable and 10 out-of-document.")
    return questions


def collect_stream(llm, messages):
    start, first, parts = time.perf_counter(), None, []
    for chunk in llm.stream(messages):
        text = rag.clean_response(chunk)
        if text:
            if first is None:
                first = time.perf_counter()
            parts.append(text)
    end = time.perf_counter()
    return "".join(parts).strip(), (first or end) - start, end - start


def parse_judge(text):
    try:
        value = json.loads(text[text.index("{"): text.rindex("}") + 1])
        score = int(value["score"])
        return score if score in (0, 1, 2) else None, str(value.get("reason", ""))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None, "unparseable judge output"


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def write_json_atomic(path, value):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--judge-model", default=rag.OLLAMA_MODEL)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise FileNotFoundError(args.pdf)
    questions = load_questions()
    pdf_hash = rag.compute_pdf_hash(args.pdf)
    rag_source_hash = hashlib.sha256(Path(rag.__file__).read_bytes()).hexdigest()
    index_configuration = {
        "pdf": pdf_hash,
        "embedding_model": rag.EMBEDDING_MODEL,
        "chunk_size": rag.CHUNK_SIZE,
        "chunk_overlap": rag.CHUNK_OVERLAP,
        "rag_source": rag_source_hash,
    }
    index_signature = hashlib.sha256(
        json.dumps(index_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    info = rag.setup_rag_pipeline(
        args.pdf,
        persist_dir=ROOT / "chroma_db" / "benchmark_indexes" / index_signature,
    )
    llm, judge = rag.get_llm(), rag.get_llm(args.judge_model)
    configuration = {
        "pdf": info["pdf_hash"],
        "questions": questions,
        "answer_model": rag.OLLAMA_MODEL,
        "judge_model": args.judge_model,
        "context_window": rag.OLLAMA_NUM_CTX,
        "embedding_model": rag.EMBEDDING_MODEL,
        "top_k": rag.TOP_K,
        "fetch_k": rag.FETCH_K,
        "mmr_lambda": rag.MMR_LAMBDA,
        "chunk_size": rag.CHUNK_SIZE,
        "chunk_overlap": rag.CHUNK_OVERLAP,
        "rag_source": rag_source_hash,
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    signature = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"benchmark_{info['pdf_hash']}_{signature}.json"
    rows = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
    completed = {r["index"] for r in rows}
    for index, item in enumerate(questions, 1):
        if index in completed:
            continue
        t0 = time.perf_counter()
        docs = rag.safe_retrieve(info["retriever"], rag.get_embeddings(), info["vectorstore"], item["question"])
        retrieval_seconds = time.perf_counter() - t0
        pages = [d.metadata.get("page") for d in docs if d.metadata.get("page") is not None]
        answer, ttft, generation_seconds = collect_stream(llm, rag.answer_prompt.format_messages(context=rag.format_context(docs), question=item["question"]))
        judge_text, _, judge_seconds = collect_stream(judge, JUDGE_PROMPT.format_messages(question=item["question"], reference=item["reference_answer"], context=rag.format_context(docs), answer=answer))
        score, reason = parse_judge(judge_text)
        rows.append({"index": index, "category": item["category"], "expected_pages": item["expected_pages"], "retrieved_pages": pages, "hit_at_8": bool(set(pages) & set(item["expected_pages"])) if item["expected_pages"] else None, "answer": answer, "judge_score": score, "judge_reason": reason, "retrieval_seconds": retrieval_seconds, "time_to_first_token_seconds": ttft, "generation_seconds": generation_seconds, "judge_seconds": judge_seconds})
        write_json_atomic(result_path, rows)
        print(f"[{index}/60] saved")
    answerable = [r for r in rows if r["hit_at_8"] is not None]
    judged = [r for r in rows if r["judge_score"] is not None]
    summary = {
        "row": "Row 0 — 4096-token context baseline",
        "configuration": {
            "model": rag.OLLAMA_MODEL,
            "judge_model": args.judge_model,
            "context_window": rag.OLLAMA_NUM_CTX,
            "top_k": rag.TOP_K,
        },
        "denominators": {
            "retrieval_hit_rate_at_8": len(answerable),
            "answer_metrics": len(judged),
            "latency": len(rows),
        },
        "metrics": {
            "retrieval_hit_rate_at_8": mean_or_none(
                [r["hit_at_8"] for r in answerable]
            ),
            "average_answer_score": mean_or_none(
                [r["judge_score"] for r in judged]
            ),
            "fully_correct_and_grounded": mean_or_none(
                [r["judge_score"] == 2 for r in judged]
            ),
        },
        "latency_seconds": {
            key: mean_or_none([r[key] for r in rows])
            for key in (
                "retrieval_seconds",
                "time_to_first_token_seconds",
                "generation_seconds",
            )
        },
        "judge_note": "Same-model judge: not independent."
        if args.judge_model == rag.OLLAMA_MODEL
        else "Separate local judge model.",
    }
    summary_path = result_path.with_name(result_path.stem + "_summary.json")
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
