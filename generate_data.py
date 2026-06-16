"""
Generate synthetic bank customer data with calculated randomization.
- 5000 customers with BNSC ID format
- Mortgage probability correlated with age + income
- Credit score correlated with missed payments, DTI, bankruptcy
- Variable rate loans more realistic
- Mortgage renewal dates within 12 months for ~15% of mortgage holders
Run once before launching app: python generate_data.py
"""

import json
import random
import os
from datetime import datetime, timedelta, date
from faker import Faker
import numpy as np

fake = Faker("en_CA")
random.seed(42)
np.random.seed(42)

# ── helpers ───────────────────────────────────────────────────────────────────

def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

# ── employment map ────────────────────────────────────────────────────────────

EMPLOYMENT_MAP = {
    "Full-Time":     ("Software Engineer", "Accountant", "Nurse", "Teacher",
                      "Marketing Manager", "Financial Analyst", "Engineer",
                      "Sales Manager", "HR Manager", "Physician",
                      "Relationship Manager", "Branch Manager", "Lawyer"),
    "Part-Time":     ("Retail Associate", "Barista", "Tutor", "Delivery Driver",
                      "Teaching Assistant", "Customer Service Rep"),
    "Self-Employed": ("Consultant", "Freelancer", "Business Owner", "Contractor",
                      "Real Estate Agent", "Financial Advisor"),
    "Unemployed":    ("—",),
    "Retired":       ("—",),
}

# ── core generator ────────────────────────────────────────────────────────────

def make_customer(idx: int) -> dict:
    # ── Employment & income ──────────────────────────────────────────────────
    emp_status = weighted_choice(
        ["Full-Time", "Part-Time", "Self-Employed", "Unemployed", "Retired"],
        [55, 10, 15, 8, 12],
    )
    job = random.choice(EMPLOYMENT_MAP[emp_status])
    age = int(np.clip(np.random.normal(42, 14), 20, 78))

    base_income = {
        "Full-Time":     np.random.normal(85_000, 30_000),
        "Part-Time":     np.random.normal(28_000, 8_000),
        "Self-Employed": np.random.normal(70_000, 40_000),
        "Unemployed":    np.random.normal(14_000, 4_000),
        "Retired":       np.random.normal(38_000, 12_000),
    }[emp_status]
    annual_income = int(clamp(base_income, 12_000, 300_000))

    years_at_employer = (
        0 if emp_status in ("Unemployed", "Retired")
        else clamp(int(np.random.exponential(4)), 0, 35)
    )

    # ── Credit profile (calculated) ──────────────────────────────────────────
    # Higher income / stable employment → better credit
    income_factor   = clamp((annual_income - 30_000) / 170_000, 0, 1)
    emp_factor      = {"Full-Time": 1.0, "Self-Employed": 0.8,
                       "Retired": 0.75, "Part-Time": 0.5, "Unemployed": 0.2}[emp_status]
    credit_base     = 580 + income_factor * 180 + emp_factor * 60
    credit_score    = int(clamp(np.random.normal(credit_base, 60), 300, 850))

    # Missed payments inversely correlated with credit score
    if credit_score >= 750:
        late_weights = [85, 10, 4, 1, 0, 0]
    elif credit_score >= 680:
        late_weights = [65, 20, 10, 4, 1, 0]
    elif credit_score >= 620:
        late_weights = [40, 25, 18, 10, 5, 2]
    else:
        late_weights = [20, 25, 22, 18, 10, 5]
    num_late_payments = weighted_choice([0, 1, 2, 3, 4, 5], late_weights)
    months_since_delinquency = (
        None if num_late_payments == 0
        else random.randint(1, 84)
    )

    # Bankruptcy inversely correlated with credit score
    bankruptcy_prob = max(0.01, min(0.25, (700 - credit_score) / 700 * 0.3))
    bankruptcy = random.random() < bankruptcy_prob

    # If bankruptcy, lower credit score further
    if bankruptcy:
        credit_score = int(clamp(credit_score - random.randint(80, 150), 300, 620))

    # ── Debt & accounts ──────────────────────────────────────────────────────
    total_debt   = int(clamp(np.random.exponential(45_000), 0, 500_000))
    num_open_accounts = random.randint(1, 12)
    checking     = clamp(int(np.random.exponential(8_000)), 100, 150_000)
    savings      = clamp(int(np.random.exponential(25_000)), 0, 500_000)
    investments  = (
        0 if age < 25
        else clamp(int(np.random.exponential(50_000)), 0, 1_000_000)
    )
    monthly_expenses = int(clamp(annual_income / 12 * np.random.uniform(0.35, 0.85), 500, 12_000))

    debt_to_income     = round(total_debt / max(annual_income, 1), 4)
    payment_to_income  = round((total_debt * 0.015) / max(annual_income / 12, 1), 4)

    # ── Risk score ───────────────────────────────────────────────────────────
    score  = (credit_score - 300) / 550 * 40
    score += max(0, (1 - debt_to_income)) * 20
    score += emp_factor * 15
    score += (0 if bankruptcy else 10)
    score += clamp((annual_income - 20_000) / 180_000 * 15, 0, 15)
    score  = clamp(score, 0, 100)
    risk_tier = "Low" if score >= 72 else "Medium" if score >= 48 else "High"

    # ── Loans (calculated randomization) ────────────────────────────────────
    today = date.today()

    # Mortgage probability: higher for 30-60yo with income > 55k, stable employment
    mortgage_age_factor    = 1.0 if 30 <= age <= 60 else (0.4 if age < 30 else 0.6)
    mortgage_income_factor = clamp((annual_income - 40_000) / 160_000, 0, 1)
    mortgage_emp_factor    = 1.0 if emp_status == "Full-Time" else 0.6 if emp_status in ("Self-Employed", "Retired") else 0.2
    mortgage_prob          = mortgage_age_factor * mortgage_income_factor * mortgage_emp_factor * 0.55

    # Auto loan probability: 25-65yo, any stable income
    auto_prob = 0.35 if emp_status != "Unemployed" and age >= 22 else 0.1

    # Line of credit: correlated with income
    loc_prob = clamp(annual_income / 200_000 * 0.45, 0.05, 0.45)

    # Personal loan: higher for lower credit scores (more need)
    personal_prob = 0.3 if credit_score < 680 else 0.2

    loans = []

    # Mortgage
    if random.random() < mortgage_prob:
        amount = random.randint(200_000, 900_000)
        # ~60% variable, ~40% fixed — variable tracks prime rate
        is_variable = random.random() < 0.60
        rate = round(random.uniform(4.5, 6.5) if is_variable else random.uniform(4.2, 6.0), 2)
        loan_type = "Mortgage"

        # Term: 5-year terms common in Canada
        # ~15% of mortgage holders renewing within 12 months
        if random.random() < 0.15:
            # Renewing soon
            start = today - timedelta(days=random.randint(4 * 365, 5 * 365 - 30))
            end   = today + timedelta(days=random.randint(30, 365))
        else:
            start = fake.date_between(start_date="-10y", end_date="-1y")
            end_year = start.year + random.choice([5, 10, 25])
            try:
                end = date(end_year, start.month, start.day)
            except ValueError:
                end = date(end_year, start.month, 28)

        duration_months = max(12, int((end - start).days / 30))
        monthly_payment = round(
            amount * (rate / 100 / 12) / (1 - (1 + rate / 100 / 12) ** -duration_months), 2
        )
        missed = random.randint(0, 2) if num_late_payments > 0 else 0
        loans.append({
            "type": loan_type,
            "amount": amount,
            "rate": rate,
            "is_variable": is_variable,
            "status": "Active" if end > today else "Closed",
            "monthly_payment": monthly_payment,
            "start_date": str(start),
            "end_date": str(end),
            "missed_payments": missed,
        })

    # Auto loan
    if random.random() < auto_prob:
        amount = random.randint(10_000, 65_000)
        rate   = round(random.uniform(5.9, 12.9), 2)
        start  = fake.date_between(start_date="-6y", end_date="-6m")
        end    = start + timedelta(days=30 * 72)
        monthly_payment = round(
            amount * (rate / 100 / 12) / (1 - (1 + rate / 100 / 12) ** -72), 2
        )
        missed = random.randint(0, 1) if num_late_payments > 0 else 0
        loans.append({
            "type": "Auto",
            "amount": amount,
            "rate": rate,
            "is_variable": False,
            "status": "Active" if end > today else "Closed",
            "monthly_payment": monthly_payment,
            "start_date": str(start),
            "end_date": str(end),
            "missed_payments": missed,
        })

    # Line of Credit (variable rate)
    if random.random() < loc_prob:
        amount = random.randint(5_000, 50_000)
        rate   = round(random.uniform(7.5, 13.5), 2)
        start  = fake.date_between(start_date="-5y", end_date="-3m")
        end    = start + timedelta(days=365)
        monthly_payment = round(amount * (rate / 100 / 12), 2)
        missed = random.randint(0, 2) if num_late_payments > 0 else 0
        loans.append({
            "type": "Line of Credit",
            "amount": amount,
            "rate": rate,
            "is_variable": True,
            "status": "Active",
            "monthly_payment": monthly_payment,
            "start_date": str(start),
            "end_date": str(end),
            "missed_payments": missed,
        })

    # Personal loan
    if random.random() < personal_prob:
        amount = random.randint(3_000, 50_000)
        rate   = round(random.uniform(8.9, 18.9), 2)
        start  = fake.date_between(start_date="-5y", end_date="-6m")
        end    = start + timedelta(days=30 * 60)
        monthly_payment = round(
            amount * (rate / 100 / 12) / (1 - (1 + rate / 100 / 12) ** -60), 2
        )
        missed = random.randint(0, 3) if num_late_payments > 0 else 0
        loans.append({
            "type": "Personal",
            "amount": amount,
            "rate": rate,
            "is_variable": False,
            "status": "Active" if end > today else "Closed",
            "monthly_payment": monthly_payment,
            "start_date": str(start),
            "end_date": str(end),
            "missed_payments": missed,
        })

    # ── Transactions (last 6 months) ─────────────────────────────────────────
    txn_categories = [
        "Groceries", "Rent/Mortgage", "Utilities", "Dining",
        "Entertainment", "Transport", "Healthcare", "Shopping",
        "Insurance", "Transfer", "Payroll", "Investment",
        "PAC Investment", "GIC Interest Payment",
    ]
    transactions = []
    now = datetime.today()
    for _ in range(random.randint(30, 120)):
        days_ago = random.randint(0, 180)
        txn_date = now - timedelta(days=days_ago)
        cat      = random.choice(txn_categories)
        is_credit = cat in ("Payroll", "Transfer", "Investment",
                             "PAC Investment", "GIC Interest Payment")
        max_amt  = 4_000 if cat == "Rent/Mortgage" else 500
        amount   = round(random.uniform(5, max_amt), 2)
        transactions.append({
            "date": txn_date.strftime("%Y-%m-%d"),
            "category": cat,
            "amount": amount if is_credit else -amount,
            "description": (
                f"{cat} — Direct Deposit" if is_credit
                else f"{cat} — {fake.company()}"
            ),
        })
    transactions.sort(key=lambda x: x["date"], reverse=True)

    return {
        "id":                           f"BNSC{idx:07d}",
        "name":                         fake.name(),
        "age":                          age,
        "gender":                       random.choice(["Male", "Female", "Non-binary", "Unspecified"]),
        "email":                        f"bnsc{idx:07d}@example.com",
        "phone":                        fake.phone_number(),
        "address":                      fake.address().replace("\n", ", "),
        "member_since":                 str(fake.date_between(start_date="-15y", end_date="-6m")),
        "employment_status":            emp_status,
        "job_title":                    job,
        "employer":                     fake.company() if emp_status not in ("Unemployed", "Retired") else "—",
        "years_at_employer":            years_at_employer,
        "annual_income":                annual_income,
        "monthly_expenses":             monthly_expenses,
        "credit_score":                 credit_score,
        "num_late_payments":            num_late_payments,
        "months_since_last_delinquency":months_since_delinquency,
        "total_debt":                   total_debt,
        "debt_to_income_ratio":         debt_to_income,
        "payment_to_income_ratio":      payment_to_income,
        "num_open_accounts":            num_open_accounts,
        "bankruptcy_history":           bankruptcy,
        "checking_balance":             checking,
        "savings_balance":              savings,
        "investment_balance":           investments,
        "total_assets":                 checking + savings + investments,
        "loans":                        loans,
        "transactions":                 transactions,
        "risk_tier":                    risk_tier,
        "risk_score":                   round(score, 1),
    }


def build_customer_document(c: dict) -> str:
    active = [l for l in c["loans"] if l["status"] == "Active"]
    closed = [l for l in c["loans"] if l["status"] == "Closed"]
    late_info = (
        "No history of late payments."
        if c["num_late_payments"] == 0
        else f"{c['num_late_payments']} late payment(s), most recent {c['months_since_last_delinquency']} months ago."
    )
    loan_detail = ""
    for l in active:
        loan_detail += (
            f"\n  - {l['type']} loan: ${l['amount']:,} at {l['rate']}% APR "
            f"({'variable' if l.get('is_variable') else 'fixed'}), "
            f"monthly ${l['monthly_payment']:,}, ends {l['end_date']}, "
            f"missed: {l['missed_payments']}."
        )
    return f"""
CUSTOMER PROFILE: {c['name']} (ID: {c['id']})
Age: {c['age']} | Gender: {c['gender']} | Member since: {c['member_since']}
Contact: {c['email']} | {c['phone']}

EMPLOYMENT & INCOME
Employment: {c['employment_status']} — {c['job_title']} at {c['employer']}
Years with employer: {c['years_at_employer']}
Annual income: ${c['annual_income']:,} | Monthly expenses: ${c['monthly_expenses']:,}

FINANCIAL HEALTH
Credit score: {c['credit_score']} ({'Excellent' if c['credit_score'] >= 750 else 'Good' if c['credit_score'] >= 700 else 'Fair' if c['credit_score'] >= 650 else 'Poor'})
Total debt: ${c['total_debt']:,} | DTI: {c['debt_to_income_ratio']:.2%}
Late payments: {late_info}
Bankruptcy: {'Yes' if c['bankruptcy_history'] else 'No'}

ACCOUNTS
Checking: ${c['checking_balance']:,} | Savings: ${c['savings_balance']:,}
Investments: ${c['investment_balance']:,} | Total assets: ${c['total_assets']:,}

LOANS (Active: {len(active)}, Closed: {len(closed)}){loan_detail if loan_detail else chr(10) + '  No active loans.'}

RISK: {c['risk_tier']} | Score: {c['risk_score']}/100
""".strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)
    n = 5000

    print(f"Generating {n} synthetic customers (calculated randomization)...")
    customers = [make_customer(i + 1) for i in range(n)]

    # Stats
    has_mortgage    = sum(1 for c in customers if any(l["type"] == "Mortgage" and l["status"] == "Active" for l in c["loans"]))
    has_variable    = sum(1 for c in customers if any(l.get("is_variable") and l["status"] == "Active" for l in c["loans"]))
    has_missed      = sum(1 for c in customers if c["num_late_payments"] > 0)
    renewing_soon   = sum(1 for c in customers for l in c["loans"]
                          if l["type"] == "Mortgage" and l["status"] == "Active"
                          and 0 < (datetime.strptime(l["end_date"], "%Y-%m-%d").date() - date.today()).days <= 365)
    high_dti        = sum(1 for c in customers if c["debt_to_income_ratio"] > 0.40)

    print(f"  Mortgages (active):      {has_mortgage:,} ({has_mortgage/n:.0%})")
    print(f"  Variable-rate loans:     {has_variable:,} ({has_variable/n:.0%})")
    print(f"  Missed payments > 0:     {has_missed:,} ({has_missed/n:.0%})")
    print(f"  Mortgage renewing ≤12mo: {renewing_soon:,}")
    print(f"  DTI > 40%:               {high_dti:,} ({high_dti/n:.0%})")

    with open("data/customers.json", "w") as f:
        json.dump(customers, f, indent=2)
    print(f"\n  → {n} customers saved to data/customers.json")
    print("\nDone. Now delete data/chroma_db/ and restart the app.")
    print("The app will re-index all customers on first launch.")


if __name__ == "__main__":
    main()
