from pathlib import Path
import re
import json
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT / "data" / "policies"
RAW_DIR = PROJECT / "data" / "raw"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT))
import mock_llm

# 1. Load policy documents
documents = []

for path in sorted(POLICY_DIR.glob("*.txt")):
    documents.append({
        "source": path.name,
        "text": path.read_text(encoding="utf-8")
    })

if not documents:
    raise FileNotFoundError("No policy documents found.")

# 2. Split documents into chunks
def split_chunks(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

chunks = []

for document in documents:
    for chunk in split_chunks(document["text"]):
        chunks.append({
            "source": document["source"],
            "text": chunk
        })

# 3. Build TF-IDF index
chunk_texts = [item["text"] for item in chunks]

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(chunk_texts)

print("POLICY RAG INDEX")
print("Documents:", len(documents))
print("Chunks:", len(chunks))
print("TF-IDF matrix:", matrix.shape)

# 4. Retrieve relevant policy chunks
def retrieve_policy(question, top_k=3):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked = scores.argsort()[::-1][:top_k]

    return [
        {
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
            "score": float(scores[i])
        }
        for i in ranked
    ]

# 5. Apply retrieval guardrail and ask offline LLM
def answer_policy_question(question, threshold=0.15):
    results = retrieve_policy(question)

    if not results or results[0]["score"] < threshold:
        return {
            "status": "NO_RELEVANT_POLICY",
            "answer": "I could not find a relevant answer in the policy documents.",
            "sources": []
        }

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT. Answer only from the supplied policy context. Do not invent policy rules."
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        }
    ]

    response = mock_llm.chat(messages)
    answer = response.choices[0].message.content

    return {
        "status": "POLICY_FOUND",
        "answer": answer,
        "sources": [item["source"] for item in results],
        "similarity": round(results[0]["score"], 4),
        "tokens": response.usage.total_tokens
    }

# 6. Run sample questions
questions = [
    "What is the refund rule for an order delivered after 45 minutes?",
    "How long does a failed deducted payment take to return?",
    "When should a fraud ticket be escalated?",
    "How are customer phone numbers protected?",
    "Can I get a refund for a cancelled order?",
    "What colour is the company logo?",
    "Who won the cricket final?"
]

results = []

for question in questions:
    result = answer_policy_question(question)

    results.append({
        "question": question,
        "status": result["status"],
        "answer": result["answer"],
        "sources": ", ".join(result["sources"])
    })

    print("\nQUESTION:", question)
    print("STATUS:", result["status"])
    print("ANSWER:", result["answer"])
    print("SOURCES:", result["sources"])

# 7. Save results
output_path = OUTPUT_DIR / "policy_rag_results.csv"
pd.DataFrame(results).to_csv(output_path, index=False)

print("\nPolicy RAG results saved:", output_path)



# 8. Evaluate official policy questions
evaluation_path = RAW_DIR / "policy_eval.csv"
evaluation_df = pd.read_csv(evaluation_path)

evaluation_results = []

for _, row in evaluation_df.iterrows():
    question = row["question"]
    expected_doc = str(row["expected_doc"]).strip()
    expected_fact = str(row["expected_fact"]).strip()

    result = answer_policy_question(question)

    retrieved_docs = result.get("sources", [])
    answer_text = str(result.get("answer", "")).lower()

    if expected_doc.upper() == "NONE":
        document_correct = result["status"] == "NO_RELEVANT_POLICY"
        fact_correct = document_correct
    else:
        document_correct = expected_doc in retrieved_docs
        fact_correct = expected_fact.lower() in answer_text

    evaluation_results.append({
        "question": question,
        "expected_doc": expected_doc,
        "expected_fact": expected_fact,
        "predicted_answer": result.get("answer", ""),
        "retrieved_sources": ", ".join(retrieved_docs),
        "status": result["status"],
        "document_correct": document_correct,
        "fact_correct": fact_correct,
        "overall_pass": document_correct and fact_correct
    })

evaluation_output = OUTPUT_DIR / "policy_rag_evaluation.csv"
evaluation_df_results = pd.DataFrame(evaluation_results)
evaluation_df_results.to_csv(evaluation_output, index=False)

print("\\nPOLICY RAG EVALUATION")
print("---------------------")
print("Total questions:", len(evaluation_df_results))
print("Document retrieval accuracy:",
      round(evaluation_df_results["document_correct"].mean() * 100, 2), "%")
print("Fact accuracy:",
      round(evaluation_df_results["fact_correct"].mean() * 100, 2), "%")
print("Overall accuracy:",
      round(evaluation_df_results["overall_pass"].mean() * 100, 2), "%")
print("Passed:", int(evaluation_df_results["overall_pass"].sum()))
print("Failed:", int((~evaluation_df_results["overall_pass"]).sum()))
print("Saved:", evaluation_output)

print("\\nEVALUATION DETAILS:")
print(evaluation_df_results[
    ["question", "status", "document_correct", "fact_correct", "overall_pass"]
].to_string(index=False))
