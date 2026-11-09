from pathlib import Path
import sys
import json
import re
import time
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT / "data" / "raw"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import the supplied mock LLM without modifying it.
sys.path.insert(0, str(PROJECT))
import mock_llm

ALLOWED_CATEGORIES = mock_llm.CATS

# 1. Load support tickets
tickets = pd.read_csv(RAW_DIR / "tickets.csv")
print("Total tickets:", len(tickets))

# 2. Privacy redaction
def redact_personal_data(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[EMAIL]', text
    )
    text = re.sub(
        r'\b(?:\+?\d[\d\s().-]{7,}\d)\b',
        '[PHONE]', text
    )
    text = re.sub(
        r'\bORD-\d+\b',
        '[ORDER_ID]', text
    )
    return text

tickets["text_redacted"] = tickets["text"].apply(
    redact_personal_data
)

# 3. Classification response parser
def parse_classification(response_text):
    try:
        result = json.loads(response_text)
        category = result.get("category")
        reason = result.get("reason", "")

        if category not in ALLOWED_CATEGORIES:
            return "technical", "Invalid category fallback"

        return category, reason

    except (json.JSONDecodeError, AttributeError, TypeError):
        return "technical", "Invalid response fallback"

# 4. Few-shot examples
EXAMPLES = """
EXAMPLES:
Message: My payment was deducted twice.
Category: billing

Message: My delivery is late.
Category: delivery

Message: The app keeps crashing.
Category: technical

Message: Someone accessed my account without permission.
Category: fraud
"""

def build_prompt(message, few_shot=True):
    system_prompt = (
        "CLASSIFY each support ticket into exactly one category: "
        "billing, delivery, technical, or fraud. "
        "Return only valid JSON with category and reason."
    )

    if few_shot:
        system_prompt += "\n" + EXAMPLES

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

# 5. Retry handling and token tracking
def classify_tickets(data, few_shot=True):
    mock_llm.reset()

    records = []
    total_retries = 0
    total_tokens = 0

    for _, row in data.iterrows():
        message = row["text_redacted"]

        if not message.strip():
            records.append({
                "ticket_id": row["ticket_id"],
                "true_category": row["true_category"],
                "predicted_category": "none",
                "reason": "Empty ticket message",
                "status": "empty",
                "tokens": 0
            })
            continue

        messages = build_prompt(message, few_shot)
        success = False
        category = "technical"
        reason = "Fallback after retries"
        token_count = 0

        for attempt in range(4):
            try:
                response = mock_llm.chat(messages)
                response_text = response.choices[0].message.content
                token_count = response.usage.total_tokens

                category, reason = parse_classification(
                    response_text
                )
                success = True
                break

            except mock_llm.TransientLLMError:
                total_retries += 1
                time.sleep(0.01)

        total_tokens += token_count

        records.append({
            "ticket_id": row["ticket_id"],
            "true_category": row["true_category"],
            "predicted_category": category,
            "reason": reason,
            "status": "classified" if success else "fallback",
            "tokens": token_count
        })

    return pd.DataFrame(records), total_retries, total_tokens

# 6. Run zero-shot classification
zero_results, zero_retries, zero_tokens = classify_tickets(
    tickets, few_shot=False
)

# 7. Run few-shot classification
few_results, few_retries, few_tokens = classify_tickets(
    tickets, few_shot=True
)

# 8. Save classification results
zero_path = OUTPUT_DIR / "triage_results_zero_shot.csv"
few_path = OUTPUT_DIR / "triage_results.csv"

zero_results.to_csv(zero_path, index=False)
few_results.to_csv(few_path, index=False)

# 9. Evaluate non-empty tickets
def evaluate(results):
    valid_categories = (
        results["true_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    evaluated = results[
        valid_categories.isin(["billing", "delivery", "technical", "fraud"])
    ]

    correct = (
        evaluated["true_category"]
        == evaluated["predicted_category"]
    ).sum()

    accuracy = 100 * correct / len(evaluated)

    return round(accuracy, 2), int(correct), len(evaluated)

zero_accuracy, zero_correct, evaluated_count = evaluate(
    zero_results
)

few_accuracy, few_correct, _ = evaluate(few_results)

# 10. Budget comparison
budget = pd.DataFrame([
    {
        "method": "zero_shot",
        "accuracy_percent": zero_accuracy,
        "correct_predictions": zero_correct,
        "evaluated_tickets": evaluated_count,
        "total_tokens": zero_tokens,
        "retries": zero_retries
    },
    {
        "method": "few_shot",
        "accuracy_percent": few_accuracy,
        "correct_predictions": few_correct,
        "evaluated_tickets": evaluated_count,
        "total_tokens": few_tokens,
        "retries": few_retries
    }
])

budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
budget.to_csv(budget_path, index=False)

# 11. Print results
print("\nZERO-SHOT ACCURACY:", zero_accuracy)
print("ZERO-SHOT CORRECT:", zero_correct)
print("ZERO-SHOT TOKENS:", zero_tokens)
print("ZERO-SHOT RETRIES:", zero_retries)

print("\nFEW-SHOT ACCURACY:", few_accuracy)
print("FEW-SHOT CORRECT:", few_correct)
print("FEW-SHOT TOKENS:", few_tokens)
print("FEW-SHOT RETRIES:", few_retries)

print("\nBUDGET COMPARISON")
print(budget.to_string(index=False))

print("\nFILES SAVED")
print(zero_path)
print(few_path)
print(budget_path)
