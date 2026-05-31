from typing import Dict, List, Optional
import asyncio
import os


def _context_reference(contexts: List[str]) -> str:
    """Create a compact reference text from retrieved contexts for lexical metrics."""
    return " ".join(context.strip() for context in contexts if context and context.strip())[:8000]


def _fallback_bleu_score(answer: str, reference: str) -> Optional[float]:
    """Compute BLEU with retrieved context as reference when RAGAS has no reference answer."""
    if not reference:
        return None
    try:
        import sacrebleu
        return float(sacrebleu.sentence_bleu(answer, [reference]).score / 100)
    except Exception:
        return None


def _fallback_rouge_score(answer: str, reference: str) -> Optional[float]:
    """Compute ROUGE-L F1 with retrieved context as reference when RAGAS has no reference answer."""
    if not reference:
        return None
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, answer)["rougeL"].fmeasure)
    except Exception:
        return None

# RAGAS imports
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, ResponseRelevancy, Faithfulness, RougeScore
    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = None
except ImportError as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        detail = f": {RAGAS_IMPORT_ERROR}" if RAGAS_IMPORT_ERROR else ""
        return {"error": f"RAGAS not available{detail}"}
    
    if not answer or not answer.strip():
        return {"error": "No answer provided for evaluation"}

    if not contexts:
        contexts = [""]

    try:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0, base_url=base_url)
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", base_url=base_url)
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=_context_reference(contexts)
        )

        metrics = {
            "response_relevancy": ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            ),
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "bleu_score": BleuScore(),
            "rouge_score": RougeScore()
        }

        async def run_metrics() -> Dict[str, float]:
            scores = {}
            for metric_name, metric in metrics.items():
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores[metric_name] = float(score)
                except Exception as metric_error:
                    scores[f"{metric_name}_error"] = str(metric_error)

            reference = _context_reference(contexts)
            if "bleu_score" not in scores:
                bleu_score = _fallback_bleu_score(answer, reference)
                if bleu_score is not None:
                    scores["bleu_score"] = bleu_score
                    scores.pop("bleu_score_error", None)

            if "rouge_score" not in scores:
                rouge_score = _fallback_rouge_score(answer, reference)
                if rouge_score is not None:
                    scores["rouge_score"] = rouge_score
                    scores.pop("rouge_score_error", None)

            return scores

        try:
            return asyncio.run(run_metrics())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run_metrics())
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
