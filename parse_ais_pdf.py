import fitz
import sys
import pandas as pd
import re


def parse_pdf_to_datapoint(pdf_path: str) -> pd.DataFrame:
    print(f"Parsing {pdf_path}")

    doc = fitz.open(pdf_path)
    # Validate if it's an AIS document
    is_ais = False
    for i in range(min(2, len(doc))):
        if "Annual Information Statement" in doc[i].get_text():
            is_ais = True
            break
    
    if not is_ais:
        raise ValueError("Invalid document: Note that only Annual Information Statement (AIS) PDFs are supported.")

    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    lines = [l.strip() for l in full_text.split("\n") if l.strip()]

    features = {
        "age": 35,
        "salary_income": 0.0,
        "business_receipts": 0.0,
        "interest_from_deposit": 0.0,
        "dividend_income": 0.0,
        "mutual_fund_investments": 0.0,
        "sale_of_property": 0.0,
        "purchase_of_property": 0.0,
        "annual_credit_card_payments": 0.0,
        "annual_emi_payments": 0.0,
    }

    current_section = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # -----------------------
        # Detect AIS Sections
        # -----------------------

        if "Part B1" in line:
            current_section = "B1"
            i += 1
            continue

        if "Part B2" in line:
            current_section = "B2"
            i += 1
            continue

        if "Part B7" in line:
            current_section = "B7"
            i += 1
            continue

        # -----------------------
        # B1 - TDS Section
        # -----------------------
        if current_section == "B1" and line == "Active":
            try:
                amount_line = lines[i - 1].replace(",", "")
                if re.match(r"^\d+(\.\d+)?$", amount_line):
                    amount = float(amount_line)

                    context = " ".join(lines[max(0, i - 6):i])

                    if "194A" in context or "Interest" in context:
                        features["interest_from_deposit"] += amount
                    elif "192" in context or "Salary" in context:
                        features["salary_income"] += amount
                    else:
                        features["business_receipts"] += amount
            except:
                pass

        # -----------------------
        # B2 - Mutual Fund SFT
        # -----------------------
        if current_section == "B2" and "Sale of Mutual Fund" in line:
            try:
                potential_amounts = []
                for offset in range(1, 7):
                    if i + offset >= len(lines):
                        break

                    val = lines[i + offset].replace(",", "")
                    if re.match(r"^\d+(\.\d+)?$", val):
                        potential_amounts.append(float(val))

                if potential_amounts:
                    features["mutual_fund_investments"] += max(potential_amounts)
            except:
                pass

        # -----------------------
        # B7 - GST Section
        # -----------------------
        if current_section == "B7" and line == "Filed":
            try:
                amount_line = lines[i - 1].replace(",", "")
                if re.match(r"^\d+(\.\d+)?$", amount_line):
                    amount = float(amount_line)
                    features["annual_credit_card_payments"] += amount
            except:
                pass

        i += 1

    # -----------------------
    # Derived Features (MUST MATCH TRAINING)
    # -----------------------

    total_income = (
        features["salary_income"]
        + features["business_receipts"]
        + features["interest_from_deposit"]
        + features["dividend_income"]
    )

    features["annual_income"] = total_income

    features["total_outflow_ratio"] = (
        features["annual_emi_payments"]
        + features["annual_credit_card_payments"]
    ) / (total_income + 1)

    features["liquidity_ratio"] = (
        features["interest_from_deposit"]
        + features["dividend_income"]
        + features["mutual_fund_investments"]
    ) / (total_income + 1)

    return pd.DataFrame([features])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_ais_pdf.py <path>")
        sys.exit(1)

    df = parse_pdf_to_datapoint(sys.argv[1])
    print("\n--- Extracted Model Input ---")
    print(df.to_string(index=False))