#!/usr/bin/env python3
"""Batch evaluation for the NASA Mission Intelligence RAG system."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import llm_client
import rag_client
import ragas_evaluator


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a list of question objects")

    for item in questions:
        if not item.get("question"):
            raise ValueError("Each evaluation item must include a question")

    return questions


def summarize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean score per metric across successful evaluation results."""
    metric_values: Dict[str, List[float]] = {}

    for result in results:
        for metric_name, score in result.get("scores", {}).items():
            if isinstance(score, (int, float)):
                metric_values.setdefault(metric_name, []).append(float(score))

    return {
        metric_name: mean(values)
        for metric_name, values in metric_values.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch RAGAS evaluation for NASA Mission Intelligence")
    parser.add_argument("--questions", default="test_questions.json", help="Path to evaluation questions JSON")
    parser.add_argument("--chroma-dir", default="chroma_db_openai", help="ChromaDB directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="ChromaDB collection name")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI or Vocareum API key")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Chat model for answer generation")
    parser.add_argument("--n-results", type=int, default=3, help="Number of chunks to retrieve per question")
    parser.add_argument("--output", default="batch_evaluation_results.json", help="Path for structured results")
    args = parser.parse_args()

    if not args.openai_key:
        raise ValueError("Provide --openai-key or set OPENAI_API_KEY")

    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    questions = load_questions(args.questions)
    collection, success, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
    if not success:
        raise RuntimeError(f"Could not initialize ChromaDB collection: {error}")

    results = []
    for item in questions:
        question = item["question"]
        mission = item.get("mission")
        documents_result = rag_client.retrieve_documents(
            collection,
            question,
            n_results=args.n_results,
            mission_filter=mission
        )

        documents = []
        metadatas = []
        if documents_result and documents_result.get("documents"):
            documents = documents_result["documents"][0]
            metadatas = documents_result["metadatas"][0]

        context = rag_client.format_context(documents, metadatas)
        answer = llm_client.generate_response(args.openai_key, question, context, [], args.model)
        scores = ragas_evaluator.evaluate_response_quality(question, answer, documents)

        result = {
            "id": item.get("id"),
            "category": item.get("category"),
            "mission": mission,
            "question": question,
            "retrieved_context_count": len(documents),
            "answer": answer,
            "scores": scores
        }
        results.append(result)

        print(f"\n[{result['id']}] {question}")
        print(f"Retrieved contexts: {len(documents)}")
        print(f"Scores: {scores}")

    summary = summarize_scores(results)
    output = {
        "questions_file": args.questions,
        "collection": args.collection_name,
        "results": results,
        "aggregate_mean_scores": summary
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\nAggregate mean scores:")
    for metric_name, score in summary.items():
        print(f"- {metric_name}: {score:.3f}")
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
