"""
Generate synthetic bank customer data and build ChromaDB vector store.
Run this once before launching the app.
"""

import json
import random
import os
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

fake = Faker("en_CA")
random.seed(42)
np.random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────────────

def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

# ── synthetic customer generator ─────────────────────────────────────────────

EMPLOYMENT_MAP = {
    "Full-Time":    ("Software Engineer", "Accountant", "Nurse", "Teacher",
                     "Marketing Manager", "Financial Analyst", "Engineer",
                     "Sales Manager", "HR Manager", "Physician"),
    "Part-Time":    ("Retail Associate", "Barista", "Tutor", "Delivery Driver"),
    "Self-Employed":("Consultant", "Freelancer", "Business Owner", "Contractor"),
    "Unemployed":   ("—",),
    "Retired":      ("—",),
}

def make_customer(idx: int) -> dict:
    emp_status = weighted_choice(
        ["Full-Time", "Part-Time", "Self-Employed", "Unemployed", "Retired"],
        [55, 10, 15, 8, 12],
    )
    job = random.choice(EMPLOYMENT_MAP[emp_status])

    age = int(np.clip(np.random.normal(42, 14), 20, 78))

    base_income = {
        "Full-Time": np.random.normal(85_000, 30_000),
        "Part-Time": np.random.normal(28_000, 8_000),
        "Self-Employed": np.random.normal(70_000, 40_000),
        "Unemployed": np.random.normal(14_000, 4_000),
        "Retired": np.random.normal(38_000, 12_000),
    }[emp_status]
    annual_income = int(clamp(base_income, 12_000, 300_000))

    credit_score = int(clamp(np.random.normal(680, 90), 300, 850))

    years_at_employer = 0 if emp_status in ("Unemployed", "Retired") else clamp(
        int(np.random.exponential(4)), 0, 35
    )

    num_late_payments = weighted_choice([0, 1, 2, 3, 4, 5], [55, 18, 12, 8, 4, 3])
    months_since_delinquency = (
        None if num_late_payments == 0
        else random.randint(1, 84)
    )

    total_debt = int(clamp(np.random.exponential(45_000), 0, 500_000))
    num_open_accounts = random.randint(1, 12)
    bankruptcy = weighted_choice([False, True], [94, 6])

    checking = clamp(int(np.random.exponential(8_000)), 100, 150_000)
    savings  = clamp(int(np.random.exponential(25_000)), 0, 500_000)
    investments = (
        0 if age < 25
        else clamp(int(np.random.exponential(50_000)), 0, 1_000_000)
    )

    monthly_expenses = int(clamp(annual_income / 12 * np.random.uniform(0.35, 0.85), 500, 12_000))

    debt_to_income = round(total_debt / max(annual_income, 1), 4)
    payment_to_income = round(
        (total_debt * 0.015) / max(annual_income / 12, 1), 4
    )

    # derive risk tier
    score = 0
    score += (credit_score - 300) / 550 * 40       # 0-40
    score += max(0, (1 - debt_to_income)) * 20      # 0-20
    score += (1 if emp_status == "Full-Time" else
              0.6 if emp_status in ("Self-Employed", "Retired") else 0.3) * 15
    score += (0 if bankruptcy else 10)              # 0 or 10
    score += clamp((annual_income - 20_000) / 180_000 * 15, 0, 15)
    score = clamp(score, 0, 100)

    risk_tier = (
        "Low" if score >= 72
        else "Medium" if score >= 48
        else "High"
    )

    # loans
    num_loans = random.randint(0, 4)
    loan_types = ["Personal", "Auto", "Mortgage", "Student", "Business", "Line of Credit"]
    loans = []
    for _ in range(num_loans):
        ltype = random.choice(loan_types)
        amount = {
            "Personal": random.randint(5_000, 50_000),
            "Auto": random.randint(10_000, 60_000),
            "Mortgage": random.randint(150_000, 800_000),
            "Student": random.randint(10_000, 80_000),
            "Business": random.randint(20_000, 200_000),
            "Line of Credit": random.randint(5_000, 50_000),
        }[ltype]
        rate = round(random.uniform(3.5, 18.9), 2)
        start = fake.date_between(start_date="-10y", end_date="today")
        duration_months = {
            "Personal": 60, "Auto": 72, "Mortgage": 300, "Student": 120,
            "Business": 84, "Line of Credit": 12,
        }[ltype]
        end = start + timedelta(days=30 * duration_months)
        status = "Active" if end > datetime.today().date() else "Closed"
        missed = random.randint(0, 3) if num_late_payments > 0 else 0
        loans.append({
            "type": ltype,
            "amount": amount,
            "rate": rate,
            "status": status,
            "monthly_payment": round(amount * (rate / 100 / 12) / (1 - (1 + rate / 100 / 12) ** -duration_months), 2),
            "start_date": str(start),
            "end_date": str(end),
            "missed_payments": missed,
        })

    # transactions (last 6 months)
    txn_categories = ["Groceries", "Rent/Mortgage", "Utilities", "Dining",
                      "Entertainment", "Transport", "Healthcare", "Shopping",
                      "Insurance", "Transfer", "Payroll", "Investment"]
    transactions = []
    today = datetime.today()
    for _ in range(random.randint(30, 120)):
        days_ago = random.randint(0, 180)
        txn_date = today - timedelta(days=days_ago)
        cat = random.choice(txn_categories)
        is_credit = cat in ("Payroll", "Transfer", "Investment")
        amount = round(random.uniform(5, 4000 if cat == "Rent/Mortgage" else 500), 2)
        transactions.append({
            "date": txn_date.strftime("%Y-%m-%d"),
            "category": cat,
            "amount": amount if is_credit else -amount,
            "description": f"{cat} — {fake.company() if not is_credit else 'Direct Deposit'}",
        })
    transactions.sort(key=lambda x: x["date"], reverse=True)

    customer = {
        "id": f"C{idx:04d}",
        "name": fake.name(),
        "age": age,
        "gender": random.choice(["Male", "Female", "Non-binary"]),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.address().replace("\n", ", "),
        "member_since": str(fake.date_between(start_date="-15y", end_date="-6m")),
        "employment_status": emp_status,
        "job_title": job,
        "employer": fake.company() if emp_status not in ("Unemployed", "Retired") else "—",
        "years_at_employer": years_at_employer,
        "annual_income": annual_income,
        "monthly_expenses": monthly_expenses,
        "credit_score": credit_score,
        "num_late_payments": num_late_payments,
        "months_since_last_delinquency": months_since_delinquency,
        "total_debt": total_debt,
        "debt_to_income_ratio": debt_to_income,
        "payment_to_income_ratio": payment_to_income,
        "num_open_accounts": num_open_accounts,
        "bankruptcy_history": bankruptcy,
        "checking_balance": checking,
        "savings_balance": savings,
        "investment_balance": investments,
        "total_assets": checking + savings + investments,
        "loans": loans,
        "transactions": transactions,
        "risk_tier": risk_tier,
        "risk_score": round(score, 1),
    }
    return customer


def build_customer_document(c: dict) -> str:
    """Convert a customer record into a rich text document for RAG indexing."""
    active_loans = [l for l in c["loans"] if l["status"] == "Active"]
    closed_loans = [l for l in c["loans"] if l["status"] == "Closed"]

    late_info = (
        "No history of late payments."
        if c["num_late_payments"] == 0
        else (
            f"{c['num_late_payments']} late payment(s), "
            f"most recent {c['months_since_last_delinquency']} months ago."
        )
    )

    loan_detail = ""
    for l in active_loans:
        loan_detail += (
            f"\n  - {l['type']} loan: ${l['amount']:,} at {l['rate']}% APR, "
            f"monthly payment ${l['monthly_payment']:,}, started {l['start_date']}, "
            f"ends {l['end_date']}, missed payments: {l['missed_payments']}."
        )

    doc = f"""
CUSTOMER PROFILE: {c['name']} (ID: {c['id']})
Age: {c['age']} | Gender: {c['gender']} | Member since: {c['member_since']}
Contact: {c['email']} | {c['phone']}
Address: {c['address']}

EMPLOYMENT & INCOME
Employment: {c['employment_status']} — {c['job_title']} at {c['employer']}
Years with employer: {c['years_at_employer']}
Annual income: ${c['annual_income']:,}
Monthly expenses: ${c['monthly_expenses']:,}

FINANCIAL HEALTH
Credit score: {c['credit_score']} ({'Excellent' if c['credit_score'] >= 750 else 'Good' if c['credit_score'] >= 700 else 'Fair' if c['credit_score'] >= 650 else 'Poor'})
Total debt: ${c['total_debt']:,}
Debt-to-income ratio: {c['debt_to_income_ratio']:.2%}
Payment-to-income ratio: {c['payment_to_income_ratio']:.2%}
Open accounts: {c['num_open_accounts']}
Late payment history: {late_info}
Bankruptcy history: {'Yes' if c['bankruptcy_history'] else 'No'}

ACCOUNTS
Checking: ${c['checking_balance']:,}
Savings: ${c['savings_balance']:,}
Investments: ${c['investment_balance']:,}
Total assets: ${c['total_assets']:,}

LOANS (Active: {len(active_loans)}, Closed: {len(closed_loans)})
{loan_detail if loan_detail else '  No active loans.'}

RISK ASSESSMENT
Risk tier: {c['risk_tier']} | Risk score: {c['risk_score']}/100
"""
    return doc.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("Generating synthetic customer data...")
    customers = [make_customer(i + 1) for i in range(75)]
    with open("data/customers.json", "w") as f:
        json.dump(customers, f, indent=2)
    print(f"  → {len(customers)} customers saved to data/customers.json")

    print("Building ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    # drop existing collection if re-running
    try:
        chroma_client.delete_collection("bank_customers")
    except Exception:
        pass

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.create_collection(
        name="bank_customers",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metas = [], [], []
    for c in customers:
        doc = build_customer_document(c)
        docs.append(doc)
        ids.append(c["id"])
        metas.append({
            "name": c["name"],
            "risk_tier": c["risk_tier"],
            "credit_score": c["credit_score"],
            "annual_income": c["annual_income"],
        })

    # batch upsert
    batch = 10
    for start in range(0, len(docs), batch):
        collection.add(
            documents=docs[start:start+batch],
            ids=ids[start:start+batch],
            metadatas=metas[start:start+batch],
        )
    print(f"  → {len(docs)} customer documents indexed in ChromaDB")
    print("\nSetup complete. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
