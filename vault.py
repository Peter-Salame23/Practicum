"""
Data Vault — privacy-preserving access layer for MMA practicum CSV data.

Agents receive vault tokens (e.g. "barbara-1") and a restricted view of
non-sensitive fields. Actual financial data, account IDs, and PII are
resolved only at the UI layer, never flowing through agent context.
"""

import os
import re
import unicodedata
import pandas as pd

# ── internal state ─────────────────────────────────────────────────────────────

_vault: dict[str, dict] = {}         # token → full row dict
_id_to_token: dict[str, str] = {}    # customer_id → token
_token_to_id: dict[str, str] = {}    # token → customer_id

# ── fields restricted from agents ─────────────────────────────────────────────
# Anything financial (amounts, balances, rates), account IDs, and PII.

_RESTRICTED_PREFIXES = (
    "snc_customer_id", "snc_transaction_amount", "dst_transaction_amount",
    "snc_from_fi", "dst_fi", "snc_customer_type",
    "account_id", "payee_customer_id", "payer_customer_id",
    "payee_fi_id", "payer_fi_id",
    "transaction_amount", "total_inflow", "total_outflow", "total_internal",
    "filtered_amount", "gross_amount", "net_amount", "interest_amount",
    "fee_amount", "adjustment_amount", "average_cost", "gross_profit",
    "final_amount", "balance_cad", "avg_balance", "deposit_amount",
    "withdrawal_amount", "buy_amount", "sell_amount", "dividend_amount",
    "gic_account_amount", "gic_face_value", "gic_market_value",
    "gic_maturity_value", "gic_issue_value", "interest_total",
    "net_foreign_amt", "foreign_cash_advance_amt",
    "net_paywave_amt", "net_online_amt", "net_chip_pin_amt",
    "net_magnetic_stripe_amt", "net_department_store_amt",
    "net_grocery_amt", "net_dining_amt", "net_fuel_amt",
    "net_travel_amt", "net_daily_transit_amt", "net_pharma_amt",
    "net_health_amt", "net_automotive_amt", "net_entertainment_amt",
    "net_tv_streaming_amt", "net_professional_service_amt",
    "net_retail_service_amt", "net_home_improvement_amt",
    "net_telecom_utilities_amt", "net_merchant_category_other_spend_amt",
    "net_recurring_payment_amt", "net_apple_pay_amt", "net_apple_total_amt",
    "net_google_total_amt", "net_samsung_total_amt",
    "monthly_fee_cad", "previous_balance_cad",
    "total_amount_cad", "total_fee_cad", "liability_amount_cad",
    "asset_amount_cad", "coa_to_cad_fx_rate",
    "interest_rate_paid", "interest_bonus_rate",
    "rate_approval_level", "rate_source",
    "target_savings_amount", "target_spending_amount", "monthly_contribution",
)

_RESTRICTED_EXACT = {
    "customer_id", "customer_name",
    "annual_income",
    "snc_transaction_date_dda_fund", "snc_transaction_date_bb_fund",
    "snc_transaction_date_ip_fund",
    "counterparty_txn_key_dda", "counterparty_txn_key_bb",
    "counterparty_txn_key_gic",
    "payee_position_id_gic", "payer_position_id_gic",
    "payee_account_id_gic", "payer_account_id_gic",
    "payee_company_code_gic", "payer_company_code_gic",
    "payee_company_name_gic", "payer_company_name_gic",
    "payee_original_name_gic", "payer_original_name_gic",
    "payee_dealer_code_gic", "payer_dealer_code_gic",
    "payee_investment_code_gic", "payer_investment_code_gic",
    "payee_customer_id_gic", "payer_customer_id_gic",
    "primary_owner_customer_id_gic",
    "transaction_id_mft", "transaction_id_gic",
    "fund_account_id_mft", "investment_account_id_mft",
    "investment_account_id_gic",
    "gic_certificate_id_gic",
    "position_id_gic",
    "holding_id_hisa",
    "customer_id_hisa", "account_id_hisa",
    "customer_id_bb_d2d", "account_id_bb_d2d",
    "secondary_customer_id_priority_bb_d2d", "secondary_customer_id_other_bb_d2d",
    "payer_customer_id_mft", "payer_account_id_mft",
    "plan_number_gic",
    "financial_goals_summary", "financial_goals_completed",
    "financial_goals_incomplete",
}


def _is_restricted(field: str) -> bool:
    if field in _RESTRICTED_EXACT:
        return True
    return any(field.startswith(p) for p in _RESTRICTED_PREFIXES)


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")


# ── public API ─────────────────────────────────────────────────────────────────

def load(path: str) -> None:
    """Load CSV or xlsx into vault. Must be called once at startup.
    Accepts .xlsx (Cathy's format) or .csv (Peter's format)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Try the sheet name used in the MMA workbook
        for sheet in ("synthetic_data", "Synethic_Data", "Synthetic_Data", 0):
            try:
                df = pd.read_excel(path, sheet_name=sheet)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not find a readable sheet in {path}")
    else:
        df = pd.read_csv(path, encoding="latin-1")

    name_count: dict[str, int] = {}

    for _, row in df.iterrows():
        cust_id = str(row.get("customer_id", ""))
        name = str(row.get("customer_name", ""))
        first = name.split()[0] if name.split() else name
        slug = _slugify(first) or "customer"
        name_count[slug] = name_count.get(slug, 0) + 1
        token = f"{slug}-{name_count[slug]}"

        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        _vault[token] = row_dict
        if cust_id:
            _id_to_token[cust_id] = token
            _token_to_id[token] = cust_id

    print(f"[Vault] Loaded {len(_vault):,} records.")


def resolve(token: str) -> dict | None:
    """Return the full customer record for a vault token.
    For UI display only — never pass this to an agent."""
    return _vault.get(token)


def agent_view(token: str) -> dict | None:
    """Return only the non-sensitive fields an agent is allowed to see."""
    row = _vault.get(token)
    if row is None:
        return None
    safe = {"vault_token": token}
    for k, v in row.items():
        if not _is_restricted(k):
            safe[k] = v
    return safe


def agent_summary_text(token: str) -> str:
    """Plain-text summary suitable for embedding / agent context."""
    view = agent_view(token)
    if not view:
        return ""
    seg    = view.get("primary_segment", "Unknown")
    steps  = view.get("primacy_steps_away", "?")
    missing = view.get("missing_primacy_steps", "none")
    digital = view.get("digital_engagement_flag_30days", False)
    primacy = view.get("primacy_flag", False)
    advisor = view.get("note_advisor_name", "N/A")
    goal    = view.get("goal_purpose", "N/A")
    goal_status = view.get("completion_status", "N/A")
    target_date = view.get("target_date", "N/A")
    retire_age  = view.get("retirement_age", "N/A")

    products = []
    for prod, label in [
        ("has_open_chequing",        "Chequing"),
        ("has_open_savings",         "Savings"),
        ("has_open_registered_retirement_savings_account", "RRSP"),
        ("has_open_registered_retirement_income_fund_account", "RRIF"),
        ("has_open_registered_first_home_savings_account",    "FHSA"),
        ("has_open_registered_disability_savings_account",    "RDSP"),
        ("has_open_registered_education_savings_account",     "RESP"),
        ("has_advice_plus_plan",     "Advice+"),
        ("has_smart_investor_plan",  "Smart Investor"),
    ]:
        if view.get(prod):
            products.append(label)

    return (
        f"Vault token: {token}\n"
        f"Segment: {seg} | Steps to primacy: {steps}\n"
        f"Missing steps: {missing}\n"
        f"Digital engagement (30d): {digital} | Primacy flag: {primacy}\n"
        f"Advisor: {advisor}\n"
        f"Products: {', '.join(products) or 'None'}\n"
        f"Financial goal: {goal} ({goal_status})"
        + (f" | Target date: {target_date}" if target_date else "")
        + (f" | Retirement age: {retire_age}" if retire_age not in (None, "N/A", "") else "")
    )


def token_for_id(customer_id: str) -> str | None:
    return _id_to_token.get(str(customer_id))


def id_for_token(token: str) -> str | None:
    return _token_to_id.get(token)


def all_tokens() -> list[str]:
    return list(_vault.keys())


def search_tokens(query: str) -> list[str]:
    """Search tokens whose slug contains the query (first-name search)."""
    q = _slugify(query)
    return [t for t in _vault if q in t]


def is_loaded() -> bool:
    return bool(_vault)
