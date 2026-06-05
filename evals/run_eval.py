"""
Hallucination evaluation runner - validates the live local Groq/Pinecone engine.
Usage: python evals/run_eval.py

Requires your FastAPI server to be running at http://localhost:8000.
"""

import asyncio
import json
import sys
from pathlib import Path
import httpx

# Establish strict absolute path routing for Taal's Desktop Workspace
WORKSPACE_ROOT = Path(r"C:\Users\Taal\OneDrive\Desktop\Persona_TaalChawla")
API_URL = "http://localhost:8000/chat"


async def run_eval():
    qa_path = WORKSPACE_ROOT / "evals" / "golden_qa.json"
    if not qa_path.exists():
        print(f"ERROR: Could not locate the golden dataset at: {qa_path}")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results = []
    print(f"Starting test execution cycle across {len(qa_pairs)} evaluation targets...\n")

    async with httpx.AsyncClient(timeout=30) as client:
        for i, qa in enumerate(qa_pairs):
            try:
                response = await client.post(
                    API_URL,
                    json={"message": qa["question"], "conversation_history": []},
                )
                
                # Check for bad HTTP response states (e.g., 500 Internal Server Errors)
                if response.status_code != 200:
                    print(f"[{i+1:02d}] ERROR (Status {response.status_code}) — {qa['question'][:50]}...")
                    continue
                    
                response_data = response.json()
                answer = response_data.get("response", "").lower()
                
            except Exception as e:
                print(f"[{i+1:02d}] HTTP CONNECTION FAILED — {qa['question'][:50]}... Error: {e}")
                continue

            # Compute hit densities across required architectural keyword flags
            hits = [kw.lower() in answer for kw in qa["expected_keywords"]]
            score = sum(hits) / len(hits) if hits else 0.0
            is_hallucination = score == 0.0

            results.append({
                "question": qa["question"],
                "source": qa["source"],
                "score": round(score, 2),
                "hallucination": is_hallucination,
                "keywords_found": [kw for kw, hit in zip(qa["expected_keywords"], hits) if hit],
                "keywords_missed": [kw for kw, hit in zip(qa["expected_keywords"], hits) if not hit],
            })
            
            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"[{i+1:02d}] {status} ({score:.0%}) — {qa['question'][:60]}...")

    if not results:
        print("\n============================================================")
        print("ERROR: Zero evaluations successfully compiled. Check backend connectivity.")
        return []

    print("\n" + "="*60)
    hallucination_rate = sum(r["hallucination"] for r in results) / len(results)
    avg_score = sum(r["score"] for r in results) / len(results)
    
    print(f"Hallucination Rate : {hallucination_rate:.1%}  (target: <20%)")
    print(f"Average Score      : {avg_score:.1%}  (target: >70%)")
    print(f"Total questions    : {len(results)}")

    failed = [r for r in results if r["hallucination"]]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  - {r['question']}")
            print(f"    Missed keywords: {r['keywords_missed']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())