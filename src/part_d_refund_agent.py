from pathlib import Path
import sys
import json
import re
import pandas as pd
from datetime import datetime

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "clean"
RAW_DIR = PROJECT / "data" / "raw"
POLICY_DIR = PROJECT / "data" / "policies"
OUTPUT_DIR = PROJECT / "outputs"
STATE_DIR = PROJECT / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT))
import mock_llm

orders = pd.read_csv(DATA_DIR / "orders.csv")
scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv")

# Start fresh for a reproducible batch run.
for filename in ["agent_runs.jsonl", "refund_ledger.jsonl"]:
    path = OUTPUT_DIR / filename
    path.write_text("", encoding="utf-8")

# Load refund policy.
refund_policy = (
    POLICY_DIR / "refund_policy.txt"
).read_text(encoding="utf-8")

# Tool 1: Find an order.
def get_order(order_id, customer_id):
    matches = orders[
        orders["order_id"].astype(str).str.strip() == str(order_id).strip()
    ]

    if matches.empty:
        return {"error": "Order not found"}

    row = matches.iloc[0]

    if str(row["customer_id"]) != str(customer_id):
        return {"error": "Customer ownership verification failed"}

    delivery = row["delivery_minutes"]

    if pd.isna(delivery):
        delivery = None
    else:
        delivery = float(delivery)

    return {
        "order_id": str(row["order_id"]),
        "customer_id": str(row["customer_id"]),
        "status": str(row["status"]),
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": delivery
    }

# Tool 2: Retrieve refund policy.
def search_policy(query):
    return {
        "policy": refund_policy,
        "query": query
    }

# Tool 3: Calculate refund eligibility.
def calculate_refund(order_amount, delivery_minutes, status):
    if status != "completed":
        return {
            "refund_inr": 0,
            "decision": "DENIED",
            "reason": "Only completed orders are eligible."
        }

    if delivery_minutes is None:
        return {
            "refund_inr": 0,
            "decision": "DENIED",
            "reason": "Delivery time is unavailable."
        }

    if delivery_minutes <= 45:
        return {
            "refund_inr": 0,
            "decision": "DENIED",
            "reason": "Delivery was within 45 minutes."
        }

    amount = round(float(order_amount), 2)

    if amount > 1000:
        return {
            "refund_inr": amount,
            "decision": "ESCALATE",
            "reason": "Refund exceeds INR 1000."
        }

    return {
        "refund_inr": amount,
        "decision": "REFUND",
        "reason": "Eligible late delivery refund."
    }

# Tool 4: Simulate issuing a refund.
def issue_refund(order_id, refund_inr):
    transaction = {
        "transaction_id": f"TXN-{order_id}",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "SIMULATED"
    }

    with open(OUTPUT_DIR / "refund_ledger.jsonl", "a") as file:
        file.write(json.dumps(transaction) + "\n")

    return transaction

# Tool 5: Escalate to human approval.
def escalate_to_human(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "PENDING_HUMAN_APPROVAL"
    }

# Tool definitions supplied to the mock LLM.
TOOLS = [
    {"type": "function", "function": {"name": "get_order", "description": "Find and verify customer order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "customer_id": {"type": "string"}}, "required": ["order_id", "customer_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Retrieve refund policy", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate_refund", "description": "Calculate refund eligibility", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}, "status": {"type": "string"}}, "required": ["order_amount", "delivery_minutes", "status"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Simulate refund issue", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Escalate high-value refund", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}}
]

FUNCTIONS = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

def save_checkpoint(scenario_id, messages, outcome, refund):
    checkpoint = {
        "scenario_id": scenario_id,
        "messages": messages,
        "outcome": outcome,
        "refund_inr": refund,
        "saved_at": datetime.now().isoformat()
    }

    path = STATE_DIR / f"{scenario_id}_checkpoint.json"
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

def run_agent(scenario_id, customer_id, user_message, max_steps=6):
    # Special loop demonstration from supplied mock LLM.
    if "[LOOP]" in user_message:
        outcome = "FALLBACK"
        save_checkpoint(scenario_id, [], outcome, 0)
        return outcome, 0, "Loop guardrail activated."

    order_match = re.search(r"ORD-\d{6}", user_message)

    if not order_match:
        return "NEEDS_INFO", 0, "Order ID is missing."

    order_id = order_match.group(0)

    messages = [
        {
            "role": "system",
            "content": "AGENT. Process refunds using tools. Never bypass order ownership or refund policy."
        },
        {
            "role": "user",
            "content": f"{user_message} Customer ID: {customer_id}"
        }
    ]

    total_steps = 0
    verified_refund = 0.0
    mock_llm.reset()

    for step in range(max_steps):
        total_steps += 1

        try:
            response = mock_llm.chat(messages, tools=TOOLS)
        except mock_llm.TransientLLMError:
            # Retry a transient failure once.
            response = mock_llm.chat(messages, tools=TOOLS)

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            answer = message.content or "FALLBACK"
            outcome = answer.split(":")[0].strip().upper()

            refund_match = re.search(r"INR\s+([\d.]+)", answer)
            refund = verified_refund if outcome == "ESCALATE" else (
                float(refund_match.group(1)) if refund_match else 0
            )

            save_checkpoint(scenario_id, messages, outcome, refund)
            return outcome, refund, answer

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments
                    }
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:
            name = call.name
            arguments = call.arguments if isinstance(call.arguments, dict) else json.loads(call.arguments)

            # Never trust an LLM-provided customer ID.
            if name == "get_order":
                arguments["order_id"] = order_id
                arguments["customer_id"] = customer_id
                result = get_order(order_id, customer_id)

            # Validate policy result before the LLM can issue money.
            elif name == "search_policy":
                result = search_policy(arguments.get("query", "refund policy"))

            elif name == "calculate_refund":
                order = get_order(order_id, customer_id)

                if "error" in order:
                    result = {"error": order["error"]}
                else:
                    result = calculate_refund(
                        order["order_amount"],
                        order["delivery_minutes"],
                        order["status"]
                    )

            elif name == "issue_refund":
                order = get_order(order_id, customer_id)
                decision = calculate_refund(
                    order.get("order_amount", 0),
                    order.get("delivery_minutes"),
                    order.get("status", "")
                ) if "error" not in order else {"decision": "DENIED"}

                if decision["decision"] == "REFUND":
                    result = issue_refund(
                        order_id,
                        decision["refund_inr"]
                    )
                else:
                    result = {"error": "Refund authorization failed."}

            elif name == "escalate_to_human":
                order = get_order(order_id, customer_id)
                decision = calculate_refund(
                    order.get("order_amount", 0),
                    order.get("delivery_minutes"),
                    order.get("status", "")
                ) if "error" not in order else {"decision": "DENIED"}

                if decision["decision"] == "ESCALATE":
                    verified_refund = float(decision["refund_inr"])
                    result = escalate_to_human(
                        order_id,
                        verified_refund
                    )
                else:
                    result = {"error": "Escalation authorization failed."}

            else:
                result = {"error": "Unknown tool"}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result)
            })

    save_checkpoint(scenario_id, messages, "FALLBACK", 0)
    return "FALLBACK", 0, "Maximum agent steps reached."

# Run all scenarios.
results = []
agent_log = OUTPUT_DIR / "agent_runs.jsonl"

for _, scenario in scenarios.iterrows():
    scenario_id = str(scenario["scenario_id"])
    outcome, refund, explanation = run_agent(
        scenario_id,
        str(scenario["customer_id"]),
        str(scenario["message"])
    )

    expected = str(scenario["expected_outcome"])
    expected_refund = round(float(scenario["expected_refund_inr"]), 2)

    passed = (
        outcome == expected
        and round(refund, 2) == expected_refund
    )

    record = {
        "scenario_id": scenario_id,
        "customer_id": str(scenario["customer_id"]),
        "expected_outcome": expected,
        "actual_outcome": outcome,
        "expected_refund_inr": expected_refund,
        "actual_refund_inr": round(refund, 2),
        "passed": passed,
        "explanation": explanation
    }

    results.append(record)

    with open(agent_log, "a") as file:
        file.write(json.dumps(record) + "\n")

    print(
        scenario_id,
        "| Expected:", expected,
        "| Actual:", outcome,
        "| Refund:", round(refund, 2),
        "| PASS:", passed
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "agent_scenario_results.csv",
    index=False
)

print("\nAGENT SUMMARY")
print("Scenarios:", len(results_df))
print("Passed:", int(results_df["passed"].sum()))
print("Failed:", int((~results_df["passed"]).sum()))

print("\nFILES SAVED")
print(OUTPUT_DIR / "agent_scenario_results.csv")
print(OUTPUT_DIR / "agent_runs.jsonl")
print(OUTPUT_DIR / "refund_ledger.jsonl")
print(STATE_DIR)
