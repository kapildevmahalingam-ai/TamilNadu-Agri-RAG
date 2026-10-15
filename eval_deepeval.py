"""Optional DeepEval evaluation. Requires an OpenAI key and a populated FAISS index."""
import csv
import os
from dotenv import load_dotenv
from rag import answer, retrieve

load_dotenv()
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
except ImportError as exc:
    raise SystemExit("Install DeepEval with `pip install deepeval` before evaluation.") from exc

def main():
    limit = int(os.getenv("N", "3"))
    test_cases = []
    with open("eval_questions.csv", encoding="utf-8-sig", newline="") as f:
        questions = [row["question"].strip().strip('"') for row in csv.DictReader(f) if row.get("question")][:limit]
    for question in questions:
        retrieved = retrieve(question)
        result = answer(question)
        test_cases.append(LLMTestCase(
            input=question,
            actual_output=result["answer"],
            retrieval_context=[item["text"] for item in retrieved],
        ))
    metrics = [
        FaithfulnessMetric(threshold=0.70, model="gpt-4o-mini"),
        AnswerRelevancyMetric(threshold=0.70, model="gpt-4o-mini"),
        ContextualRelevancyMetric(threshold=0.50, model="gpt-4o-mini"),
    ]
    evaluate(test_cases=test_cases, metrics=metrics)

if __name__ == "__main__":
    main()
