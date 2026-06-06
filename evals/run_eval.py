"""
Hallucination eval — run against the live local server.
Usage: python evals/run_eval.py

Requires the FastAPI server to be running at localhost:8000.
3-second delay between requests to avoid Gemini free-tier RPM limits.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000/chat"
REQUEST_DELAY_SECONDS = 6  


async def run_eval():
    qa_path = Path(__file__).parent / "golden_qa.json"
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    print(f"Starting test execution cycle across {len(qa_pairs)} evaluation targets...\n")

    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        for i, qa in enumerate(qa_pairs):

            try:
                response = await client.post(
                    API_URL,
                    json={"message": qa["question"], "conversation_history": []},
                )

                if response.status_code != 200:
                    print(f"[{i+1:02d}] ERROR (Status {response.status_code}) — {qa['question'][:60]}...")
                    results.append({
                        "question": qa["question"],
                        "source": qa["source"],
                        "score": 0.0,
                        "hallucination": True,
                        "error": f"HTTP {response.status_code}",
                        "keywords_found": [],
                        "keywords_missed": qa["expected_keywords"],
                    })
                else:
                    answer = response.json().get("response", "").lower()
                    hits = [kw.lower() in answer for kw in qa["expected_keywords"]]
                    score = sum(hits) / len(hits)
                    is_hallucination = score == 0.0

                    status = "PASS" if score >= 0.5 else "FAIL"
                    print(f"[{i+1:02d}] {status} ({score:.0%}) — {qa['question'][:60]}...")

                    results.append({
                        "question": qa["question"],
                        "source": qa["source"],
                        "score": round(score, 2),
                        "hallucination": is_hallucination,
                        "keywords_found": [kw for kw, hit in zip(qa["expected_keywords"], hits) if hit],
                        "keywords_missed": [kw for kw, hit in zip(qa["expected_keywords"], hits) if not hit],
                    })

            except Exception as e:
                print(f"[{i+1:02d}] ERROR (Exception) — {qa['question'][:60]}... | {e}")
                results.append({
                    "question": qa["question"],
                    "source": qa["source"],
                    "score": 0.0,
                    "hallucination": True,
                    "error": str(e),
                    "keywords_found": [],
                    "keywords_missed": qa["expected_keywords"],
                })

            # Delay between requests — critical for Gemini free-tier RPM limits
            if i < len(qa_pairs) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 60)

    # Only count non-error results in metrics
    valid = [r for r in results if "error" not in r]
    error_count = len(results) - len(valid)

    if valid:
        hallucination_rate = sum(r["hallucination"] for r in valid) / len(valid)
        avg_score = sum(r["score"] for r in valid) / len(valid)
        print(f"Hallucination Rate : {hallucination_rate:.1%}  (target: <20%)")
        print(f"Average Score      : {avg_score:.1%}  (target: >70%)")
    print(f"Total questions    : {len(results)}")
    print(f"Errors (500/timeout): {error_count}")

    failed = [r for r in results if r.get("hallucination") and "error" not in r]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  - {r['question']}")
            print(f"    Missed: {r['keywords_missed']}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nError questions ({len(errors)}):")
        for r in errors:
            print(f"  - {r['question']}")
            print(f"    Error: {r['error']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())