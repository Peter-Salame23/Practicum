from __future__ import annotations

from datetime import date, datetime
import re


RULE_THRESHOLDS = {
    "critical_liquidity_months": 1.0,
    "low_liquidity_months": 3.0,
    "healthy_liquidity_min_months": 3.0,
    "healthy_liquidity_max_months": 6.0,
    "excess_liquidity_months": 6.0,
    "idle_chequing_months": 3.0,
    "very_high_idle_chequing_months": 6.0,
    "negative_surplus_ratio": 0.00,
    "narrow_surplus_ratio": 0.10,
    "healthy_surplus_ratio": 0.15,
    "strong_surplus_ratio": 0.25,
    "expense_growth_threshold": 0.15,
    "income_volatility_threshold": 0.30,
    "minimum_expense_growth_amount": 100,
    "moderate_debt_service_ratio": 0.25,
    "high_debt_service_ratio": 0.40,
    "very_high_debt_service_ratio": 0.44,
    "credit_utilization_watch": 0.30,
    "credit_utilization_high": 0.70,
    "late_payment_watch": 1,
    "frequent_late_payments": 2,
    "overdraft_watch_events": 1,
    "frequent_overdraft_events": 3,
    "low_balance_events": 3,
    "near_fee_waiver_gap_fixed": 500,
    "near_fee_waiver_gap_pct": 0.10,
    "monthly_fee_watch": 1,
    "frequent_external_abm_usage": 2,
    "high_external_abm_usage": 4,
    "foreign_spend_watch_amount": 100,
    "foreign_spend_high_amount": 500,
    "frequent_usd_transactions": 3,
    "gic_maturity_window_days": 90,
    "student_age_cutoff": 23,
    "senior_age_cutoff": 60,
}


def get_adjusted_thresholds(customer):
    thresholds = RULE_THRESHOLDS.copy()

    risk_tier = str(customer.get("risk_tier", "Medium")).title()
    segment = str(customer.get("primary_segment", "")).title()

    risk_adjustments = {
        "Low": {
            "low_liquidity_months": 3.0,
            "healthy_liquidity_min_months": 3.0,
            "narrow_surplus_ratio": 0.10,
            "healthy_surplus_ratio": 0.15,
            "strong_surplus_ratio": 0.25,
            "moderate_debt_service_ratio": 0.25,
            "high_debt_service_ratio": 0.40,
        },
        "Medium": {
            "low_liquidity_months": 3.5,
            "healthy_liquidity_min_months": 3.5,
            "narrow_surplus_ratio": 0.12,
            "healthy_surplus_ratio": 0.18,
            "strong_surplus_ratio": 0.28,
            "moderate_debt_service_ratio": 0.22,
            "high_debt_service_ratio": 0.36,
        },
        "High": {
            "low_liquidity_months": 4.0,
            "healthy_liquidity_min_months": 4.0,
            "narrow_surplus_ratio": 0.15,
            "healthy_surplus_ratio": 0.20,
            "strong_surplus_ratio": 0.30,
            "moderate_debt_service_ratio": 0.20,
            "high_debt_service_ratio": 0.33,
        },
    }

    segment_adjustments = {
        "Primacy": {
            "idle_chequing_months": 4.0,
            "very_high_idle_chequing_months": 8.0,
            "foreign_spend_high_amount": 1000,
        },
        "Near Primacy": {
            "idle_chequing_months": 3.5,
            "very_high_idle_chequing_months": 7.0,
            "foreign_spend_high_amount": 750,
        },
    }

    thresholds.update(risk_adjustments.get(risk_tier, risk_adjustments["Medium"]))

    for segment_name, adjustment in segment_adjustments.items():
        if segment_name.lower() in segment.lower():
            thresholds.update(adjustment)

    return thresholds


def safe_num(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "yes", "y", "1", "active", "open"}


def _safe_text(value, default=""):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _safe_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _normalize_goal_signal(goal_purpose):
    purpose = re.sub(r"[^a-z0-9]+", "_", _safe_text(goal_purpose).lower()).strip("_")
    return f"{purpose}_goal" if purpose else None


def _goal_records(customer):
    goals = customer.get("financial_goals")
    if isinstance(goals, list) and goals:
        return [goal for goal in goals if isinstance(goal, dict)]
    single_goal = {}
    for key in (
        "goal_purpose",
        "completion_status",
        "target_savings_amount",
        "target_spending_amount",
        "target_date",
        "monthly_contribution",
        "completion_amount",
    ):
        if key in customer:
            single_goal[key] = customer.get(key)
    if single_goal:
        if single_goal.get("goal_purpose") and not single_goal.get("purpose"):
            single_goal["purpose"] = single_goal.get("goal_purpose")
        if single_goal.get("completion_status") and not single_goal.get("status"):
            single_goal["status"] = single_goal.get("completion_status")
        if single_goal.get("target_savings_amount") and not single_goal.get("target_amount"):
            single_goal["target_amount"] = single_goal.get("target_savings_amount")
        return [single_goal]
    return []


def build_customer_metrics(customer):
    annual_income = safe_num(customer.get("annual_income"))
    monthly_income = annual_income / 12 if annual_income else 0
    monthly_expenses = safe_num(customer.get("monthly_expenses"))
    checking_balance = safe_num(customer.get("checking_balance"))
    savings_balance = safe_num(customer.get("savings_balance"))
    investment_balance = safe_num(customer.get("investment_balance"))
    liquid_assets = checking_balance + savings_balance

    active_loans = [
        loan for loan in customer.get("loans", [])
        if isinstance(loan, dict) and _safe_text(loan.get("status")).lower() == "active"
    ]
    active_loan_payment = sum(safe_num(loan.get("monthly_payment")) for loan in active_loans)
    active_loan_amount = sum(safe_num(loan.get("amount")) for loan in active_loans)

    monthly_surplus_after_debt = monthly_income - monthly_expenses - active_loan_payment
    surplus_ratio_after_debt = (
        monthly_surplus_after_debt / monthly_income if monthly_income else 0
    )

    external_transfer_out_total = 0.0
    for transaction in customer.get("transactions", []):
        if not isinstance(transaction, dict):
            continue
        desc = f"{_safe_text(transaction.get('description'))} {_safe_text(transaction.get('category'))}".lower()
        if "external transfer out" in desc or "external transfer" in desc:
            external_transfer_out_total += abs(safe_num(transaction.get("amount")))

    liquidity_months = liquid_assets / monthly_expenses if monthly_expenses else 0
    chequing_liquidity_months = checking_balance / monthly_expenses if monthly_expenses else 0

    return {
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "checking_balance": checking_balance,
        "savings_balance": savings_balance,
        "investment_balance": investment_balance,
        "liquid_assets": liquid_assets,
        "liquidity_months": liquidity_months,
        "chequing_liquidity_months": chequing_liquidity_months,
        "active_loan_payment": active_loan_payment,
        "active_loan_amount": active_loan_amount,
        "monthly_surplus_after_debt": monthly_surplus_after_debt,
        "surplus_ratio_after_debt": surplus_ratio_after_debt,
        "total_assets": safe_num(customer.get("total_assets")),
        "total_debt": safe_num(customer.get("total_debt")),
        "net_asset_position": safe_num(customer.get("total_assets")) - safe_num(customer.get("total_debt")),
        "debt_to_income_ratio": safe_num(customer.get("debt_to_income_ratio")),
        "payment_to_income_ratio": safe_num(customer.get("payment_to_income_ratio")),
        "credit_score": safe_num(customer.get("credit_score")),
        "num_late_payments": int(safe_num(customer.get("num_late_payments"))),
        "external_transfer_out_total": external_transfer_out_total,
        "net_foreign_amt_30days": safe_num(customer.get("net_foreign_amt_30days")),
        "days_to_maturity_gic": safe_num(customer.get("days_to_maturity_gic")),
    }


def infer_client_signals(customer, metrics, thresholds=None):
    thresholds = thresholds or get_adjusted_thresholds(customer)
    signals = set()
    liquidity_months = metrics["liquidity_months"]
    chequing_liquidity_months = metrics["chequing_liquidity_months"]
    checking_balance = metrics["checking_balance"]
    surplus_ratio = metrics["surplus_ratio_after_debt"]
    dti = metrics["debt_to_income_ratio"]
    late_payments = metrics["num_late_payments"]

    if liquidity_months < thresholds["critical_liquidity_months"]:
        signals.add("critical_liquidity")
    if liquidity_months < thresholds["low_liquidity_months"]:
        signals.add("low_emergency_fund")
    if thresholds["healthy_liquidity_min_months"] <= liquidity_months <= thresholds["healthy_liquidity_max_months"]:
        signals.add("healthy_liquidity")
    if liquidity_months > thresholds["excess_liquidity_months"]:
        signals.add("excess_liquidity")
    if chequing_liquidity_months > thresholds["idle_chequing_months"]:
        signals.add("idle_cash_balance")
    if checking_balance >= 6000:
        signals.update({"balance_above_6000", "high_deposit_balance"})
    if surplus_ratio >= thresholds["strong_surplus_ratio"]:
        signals.update({"strong_savings_capacity", "positive_cash_flow"})
    elif surplus_ratio >= thresholds["healthy_surplus_ratio"]:
        signals.update({"healthy_cash_flow", "positive_cash_flow"})
    elif surplus_ratio < thresholds["narrow_surplus_ratio"]:
        signals.add("narrow_surplus")
    if dti >= thresholds["high_debt_service_ratio"]:
        signals.add("high_debt_pressure")
    elif dti >= thresholds["moderate_debt_service_ratio"]:
        signals.add("moderate_debt_pressure")
    if late_payments == 0:
        signals.add("clean_payment_history")
    if metrics["external_transfer_out_total"] > 1000:
        signals.update({"external_transfer_out", "relationship_leakage"})
    if _truthy(customer.get("has_active_savings")):
        signals.add("has_savings_balance")
    if _truthy(customer.get("has_smart_investor_plan")):
        signals.add("smart_investor_client")
    if _truthy(customer.get("has_pac_last_30days")):
        signals.add("active_monthly_contribution")
    if _truthy(customer.get("has_digital_engagement_last_30days")):
        signals.add("digital_banking_user")
    if metrics["net_foreign_amt_30days"] > thresholds["foreign_spend_high_amount"]:
        signals.update({"foreign_currency_activity", "usd_transactions", "fx_sensitive_client"})
    if 0 < metrics["days_to_maturity_gic"] <= thresholds["gic_maturity_window_days"]:
        signals.add("maturing_gic")
    elif metrics["days_to_maturity_gic"] > thresholds["gic_maturity_window_days"]:
        signals.add("has_gic")

    age = safe_num(customer.get("age"), default=-1)
    if age >= thresholds["senior_age_cutoff"]:
        signals.add("age_60_plus")

    for goal in _goal_records(customer):
        status = _safe_text(goal.get("status") or goal.get("completion_status")).lower()
        if status == "in progress":
            signals.add("savings_goal")
            purpose_signal = _normalize_goal_signal(goal.get("purpose") or goal.get("goal_purpose"))
            if purpose_signal:
                signals.add(purpose_signal)
            purpose_text = _safe_text(goal.get("purpose") or goal.get("goal_purpose")).lower()
            if age <= thresholds["student_age_cutoff"] or "education" in purpose_text or "student" in purpose_text:
                signals.add("student_status")

    return sorted(signals)


def generate_financial_insights(customer):
    metrics = build_customer_metrics(customer)
    thresholds = get_adjusted_thresholds(customer)
    client_signals = infer_client_signals(customer, metrics, thresholds=thresholds)
    customer_id = customer.get("id") or customer.get("customer_id")
    customer_name = customer.get("name") or customer.get("customer_name")
    insights = []

    def add_insight(
        insight_id,
        insight_type,
        title,
        severity,
        confidence,
        financial_meaning,
        advisor_action,
        insight_metrics,
        triggered_rules,
        required_signals=None,
    ):
        insights.append(
            {
                "insight_id": insight_id,
                "insight_type": insight_type,
                "title": title,
                "severity": severity,
                "confidence": confidence,
                "metrics": insight_metrics,
                "triggered_rules": triggered_rules,
                "client_signals": [signal for signal in client_signals if not required_signals or signal in required_signals],
                "financial_meaning": financial_meaning,
                "advisor_action": advisor_action,
            }
        )

    liquidity_months = metrics["liquidity_months"]
    chequing_liquidity_months = metrics["chequing_liquidity_months"]
    if liquidity_months < thresholds["critical_liquidity_months"]:
        add_insight(
            "LIQUIDITY_001",
            "Risk",
            "Critical liquidity risk",
            "High",
            "High",
            "The client has less than one month of liquid assets relative to current monthly expenses, which suggests limited emergency liquidity.",
            "Confirm immediate cash flow resilience, short-term obligations, and whether a contingency savings plan is needed before discussing longer-term products.",
            {"liquidity_months": round(liquidity_months, 2), "liquid_assets": metrics["liquid_assets"], "monthly_expenses": metrics["monthly_expenses"]},
            [f"liquidity_months < {thresholds['critical_liquidity_months']}"],
            {"critical_liquidity", "low_emergency_fund"},
        )
    elif liquidity_months < thresholds["low_liquidity_months"]:
        add_insight(
            "LIQUIDITY_002",
            "Watchlist",
            "Low emergency liquidity buffer",
            "Medium",
            "High",
            "The client has some liquid assets, but the emergency buffer is below the common three-month threshold.",
            "Review whether current cash reserves are intentional and whether the client needs a more resilient short-term savings buffer.",
            {"liquidity_months": round(liquidity_months, 2), "liquid_assets": metrics["liquid_assets"]},
            [f"{thresholds['critical_liquidity_months']} <= liquidity_months < {thresholds['low_liquidity_months']}"],
            {"low_emergency_fund"},
        )
    elif thresholds["healthy_liquidity_min_months"] <= liquidity_months <= thresholds["healthy_liquidity_max_months"]:
        add_insight(
            "LIQUIDITY_003",
            "Opportunity",
            "Healthy liquidity with savings review opportunity",
            "Low",
            "High",
            "The client maintains a healthy level of liquid assets relative to monthly expenses.",
            "Confirm the emergency-fund target, then review whether excess cash should remain liquid or be redirected toward savings, investing, or goal funding.",
            {"liquidity_months": round(liquidity_months, 2), "liquid_assets": metrics["liquid_assets"]},
            [f"{thresholds['healthy_liquidity_min_months']} <= liquidity_months <= {thresholds['healthy_liquidity_max_months']}"],
            {"healthy_liquidity"},
        )
    if chequing_liquidity_months > thresholds["idle_chequing_months"]:
        add_insight(
            "LIQUIDITY_004",
            "Opportunity",
            "Idle chequing cash opportunity",
            "Low",
            "Medium",
            "A large amount of spending liquidity appears to be sitting in chequing relative to monthly expenses.",
            "Confirm day-to-day cash needs and upcoming payments before discussing transfers to savings, GICs, or investment products.",
            {"chequing_liquidity_months": round(chequing_liquidity_months, 2), "checking_balance": metrics["checking_balance"]},
            [f"chequing_liquidity_months > {thresholds['idle_chequing_months']}"],
            {"idle_cash_balance"},
        )

    surplus_ratio = metrics["surplus_ratio_after_debt"]
    if surplus_ratio < thresholds["negative_surplus_ratio"]:
        add_insight(
            "CASHFLOW_001",
            "Risk",
            "Negative monthly cash flow after debt service",
            "High",
            "High",
            "After expenses and active debt payments, the client appears to be operating at a monthly deficit.",
            "Validate whether expenses are temporary or recurring, and discuss budget pressure before recommending additional commitments.",
            {"monthly_surplus_after_debt": metrics["monthly_surplus_after_debt"], "surplus_ratio_after_debt": round(surplus_ratio, 3)},
            [f"surplus_ratio_after_debt < {thresholds['negative_surplus_ratio']}"],
            {"narrow_surplus"},
        )
    elif surplus_ratio < thresholds["narrow_surplus_ratio"]:
        add_insight(
            "CASHFLOW_002",
            "Watchlist",
            "Narrow monthly surplus after debt service",
            "Medium",
            "High",
            "The client still has positive cash flow, but monthly flexibility after expenses and debt payments is limited.",
            "Review contribution levels, discretionary spending, and upcoming obligations before positioning new savings or credit recommendations.",
            {"monthly_surplus_after_debt": metrics["monthly_surplus_after_debt"], "surplus_ratio_after_debt": round(surplus_ratio, 3)},
            [f"{thresholds['negative_surplus_ratio']} <= surplus_ratio_after_debt < {thresholds['narrow_surplus_ratio']}"],
            {"narrow_surplus"},
        )
    elif thresholds["healthy_surplus_ratio"] <= surplus_ratio < thresholds["strong_surplus_ratio"]:
        add_insight(
            "CASHFLOW_003",
            "Opportunity",
            "Healthy monthly surplus after debt service",
            "Low",
            "Medium",
            "The client appears to have reasonable monthly flexibility after covering expenses and current debt obligations.",
            "Review whether surplus cash is being directed efficiently toward goals, savings buffers, or investment opportunities.",
            {"monthly_surplus_after_debt": metrics["monthly_surplus_after_debt"], "surplus_ratio_after_debt": round(surplus_ratio, 3)},
            [f"{thresholds['healthy_surplus_ratio']} <= surplus_ratio_after_debt < {thresholds['strong_surplus_ratio']}"],
            {"healthy_cash_flow", "positive_cash_flow"},
        )
    else:
        add_insight(
            "CASHFLOW_004",
            "Opportunity",
            "Strong savings capacity after debt service",
            "Low",
            "High",
            "The client appears to retain a strong monthly surplus after expenses and active debt payments.",
            "Confirm the surplus is sustainable, then review savings, investment, or goal-acceleration options aligned with liquidity needs.",
            {"monthly_surplus_after_debt": metrics["monthly_surplus_after_debt"], "surplus_ratio_after_debt": round(surplus_ratio, 3)},
            [f"surplus_ratio_after_debt >= {thresholds['strong_surplus_ratio']}"],
            {"strong_savings_capacity", "positive_cash_flow"},
        )

    dti = metrics["debt_to_income_ratio"]
    pti = metrics["payment_to_income_ratio"]
    if dti >= thresholds["high_debt_service_ratio"]:
        add_insight(
            "DEBT_001",
            "Risk",
            "High debt pressure relative to income",
            "High",
            "High",
            "Debt obligations appear elevated relative to income, which may reduce flexibility for new borrowing or additional commitments.",
            "Avoid assuming new credit is appropriate; confirm affordability, repayment capacity, and client need before discussing additional borrowing.",
            {"debt_to_income_ratio": dti, "payment_to_income_ratio": pti, "total_debt": metrics["total_debt"]},
            [f"debt_to_income_ratio >= {thresholds['high_debt_service_ratio']}"],
            {"high_debt_pressure"},
        )
    elif dti >= thresholds["moderate_debt_service_ratio"]:
        add_insight(
            "DEBT_002",
            "Watchlist",
            "Moderate debt pressure review",
            "Medium",
            "High",
            "Debt obligations are meaningful but may still be manageable depending on cash flow stability and payment behavior.",
            "Review the mix of debt, payment burden, and near-term cash flow before positioning additional credit or refinancing conversations.",
            {"debt_to_income_ratio": dti, "payment_to_income_ratio": pti, "total_debt": metrics["total_debt"]},
            [f"{thresholds['moderate_debt_service_ratio']} <= debt_to_income_ratio < {thresholds['high_debt_service_ratio']}"],
            {"moderate_debt_pressure"},
        )
    elif pti < 0.10 and metrics["num_late_payments"] == 0:
        add_insight(
            "DEBT_003",
            "Strength",
            "Debt burden appears manageable",
            "Low",
            "Medium",
            "Current payment burden appears modest relative to income and the payment record looks clean.",
            "Use this as a supporting signal only; still verify affordability and client need before discussing any borrowing solution.",
            {"payment_to_income_ratio": pti, "num_late_payments": metrics["num_late_payments"]},
            ["payment_to_income_ratio < 0.10 and num_late_payments == 0"],
            {"clean_payment_history"},
        )

    if metrics["external_transfer_out_total"] > 1000:
        add_insight(
            "REL_001",
            "Opportunity",
            "Relationship leakage review",
            "Medium",
            "Medium",
            "Meaningful funds are moving out of the current relationship through external transfers.",
            "Ask what the outgoing transfers are funding. They may indicate external savings, investing, bill funding, or another primary financial relationship.",
            {"external_transfer_out_total": metrics["external_transfer_out_total"]},
            ["external_transfer_out_total > 1000"],
            {"external_transfer_out", "relationship_leakage"},
        )

    if _truthy(customer.get("has_active_savings")) and liquidity_months >= thresholds["healthy_liquidity_min_months"]:
        add_insight(
            "SAV_001",
            "Opportunity",
            "Savings balance review opportunity",
            "Low",
            "Medium",
            "The client has an active savings relationship and appears to maintain at least a basic liquidity buffer.",
            "Confirm target liquidity needs, then review whether cash placement across chequing, savings, and term products still matches the client’s objectives.",
            {"liquidity_months": round(liquidity_months, 2), "savings_balance": metrics["savings_balance"]},
            [f"has_active_savings and liquidity_months >= {thresholds['healthy_liquidity_min_months']}"],
            {"has_savings_balance", "healthy_liquidity"},
        )
    if 0 < metrics["days_to_maturity_gic"] <= thresholds["gic_maturity_window_days"]:
        add_insight(
            "SAV_002",
            "Opportunity",
            "Upcoming GIC maturity review",
            "Medium",
            "High",
            "A GIC appears to be approaching maturity, which may create a time-sensitive placement decision.",
            "Discuss upcoming liquidity needs, renewal options, and whether the maturity should remain in term savings or be redirected to another goal.",
            {"days_to_maturity_gic": metrics["days_to_maturity_gic"]},
            [f"days_to_maturity_gic <= {thresholds['gic_maturity_window_days']}"],
            {"maturing_gic"},
        )
    elif metrics["days_to_maturity_gic"] > thresholds["gic_maturity_window_days"]:
        add_insight(
            "SAV_003",
            "Opportunity",
            "General GIC allocation review",
            "Low",
            "Medium",
            "The client appears to hold a term product, but there is no immediate maturity event.",
            "Confirm that the current GIC allocation still fits liquidity needs, time horizon, and goal priorities.",
            {"days_to_maturity_gic": metrics["days_to_maturity_gic"]},
            [f"days_to_maturity_gic > {thresholds['gic_maturity_window_days']}"],
            {"has_gic"},
        )

    if metrics["net_foreign_amt_30days"] > thresholds["foreign_spend_high_amount"]:
        add_insight(
            "FX_001",
            "Opportunity",
            "Foreign-currency activity review",
            "Medium",
            "Medium",
            "The client shows meaningful recent foreign-currency or cross-border transaction activity.",
            "Ask whether the activity is recurring. The client may have USD, foreign purchase, cross-border payment, travel, or investment-related FX needs.",
            {"net_foreign_amt_30days": metrics["net_foreign_amt_30days"]},
            [f"net_foreign_amt_30days > {thresholds['foreign_spend_high_amount']}"],
            {"foreign_currency_activity", "usd_transactions", "fx_sensitive_client"},
        )

    for index, goal in enumerate(_goal_records(customer), start=1):
        status = _safe_text(goal.get("status") or goal.get("completion_status")).lower()
        if status == "in progress":
            purpose = _safe_text(goal.get("purpose") or goal.get("goal_purpose"), "Financial goal")
            target_amount = safe_num(goal.get("target_amount") or goal.get("target_savings_amount"))
            monthly_contribution = safe_num(goal.get("monthly_contribution"))
            target_date = _safe_text(goal.get("target_date"))
            add_insight(
                f"GOAL_{index:03d}",
                "Opportunity",
                f"{purpose} funding review",
                "Medium",
                "Medium",
                "The client has an active financial goal that may require ongoing contribution discipline and timeline review.",
                "Review whether the goal contribution pace is aligned with the target date, liquidity needs, and competing priorities.",
                {
                    "goal_purpose": purpose,
                    "target_amount": target_amount,
                    "monthly_contribution": monthly_contribution,
                    "target_date": target_date,
                },
                ["financial goal status == In Progress"],
                {"savings_goal", _normalize_goal_signal(purpose) or ""},
            )

    data_quality_rules = []
    missing_fields = [
        field for field in ("annual_income", "monthly_expenses", "total_assets", "total_debt")
        if customer.get(field) in (None, "")
    ]
    if missing_fields:
        data_quality_rules.append(f"Missing core fields: {', '.join(missing_fields)}")

    business_effective_date = _parse_date(customer.get("business_effective_date"))
    future_txn_count = 0
    if business_effective_date:
        for transaction in customer.get("transactions", []):
            if not isinstance(transaction, dict):
                continue
            txn_date = _parse_date(transaction.get("date"))
            if txn_date and (txn_date - business_effective_date).days > 400:
                future_txn_count += 1
    if future_txn_count:
        data_quality_rules.append("Some transaction dates appear far after the business effective date")

    age = safe_num(customer.get("age"), default=-1)
    if age >= 0 and age < RULE_THRESHOLDS["student_age_cutoff"] and _truthy(customer.get("has_open_registered_retirement_income_fund_account")):
        data_quality_rules.append("Age and product mix may be inconsistent")

    if data_quality_rules:
        add_insight(
            "DATA_001",
            "Data Quality",
            "Source-data verification recommended",
            "Medium",
            "Low",
            "Some fields appear incomplete or potentially inconsistent, so the dashboard should be treated as decision support rather than a final source of truth.",
            "Use the dashboard as decision support and verify source-system details before relying on the insight.",
            {"missing_fields": missing_fields, "future_transaction_count": future_txn_count},
            data_quality_rules,
        )

    return {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "metrics": metrics,
        "thresholds": thresholds,
        "client_signals": client_signals,
        "insights": insights,
    }
