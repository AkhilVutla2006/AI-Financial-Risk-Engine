import pandas as pd
import numpy as np
import os

def synthesize_financial_data(
    num_samples: int = 10000,
    output_path: str = 'dataset/financial_risk_data.csv'
):

    np.random.seed(42)

    # 1️⃣ Demographics
    age = np.random.randint(21, 65, size=num_samples)

    # 2️⃣ Income (AIS-scale realistic)
    salary_income = np.random.exponential(scale=1_200_000, size=num_samples)
    salary_income = np.clip(salary_income, 0, 20_000_000)

    has_business = np.random.choice([0,1], size=num_samples, p=[0.6,0.4])
    business_receipts = (
        np.random.exponential(scale=2_000_000, size=num_samples)
        * has_business
    )
    business_receipts = np.clip(business_receipts, 0, 50_000_000)

    interest_from_deposit = np.random.exponential(scale=100_000, size=num_samples)
    dividend_income = np.random.exponential(scale=50_000, size=num_samples)
    mutual_fund_investments = np.random.exponential(scale=300_000, size=num_samples)

    # 3️⃣ Capital transactions
    sale_of_property = np.random.choice([0,1], size=num_samples, p=[0.9,0.1]) \
                        * np.random.uniform(2_000_000, 25_000_000, size=num_samples)

    purchase_of_property = np.random.choice([0,1], size=num_samples, p=[0.85,0.15]) \
                            * np.random.uniform(2_000_000, 30_000_000, size=num_samples)

    # 4️⃣ Compute Total Income
    total_income = (
        salary_income
        + business_receipts
        + interest_from_deposit
        + dividend_income
    )

    # 5️⃣ Debt behavior ratios (core driver)
    emi_ratio = np.random.beta(2, 5, size=num_samples)
    cc_ratio = np.random.beta(2, 6, size=num_samples)

    annual_emi_payments = emi_ratio * total_income
    annual_credit_card_payments = cc_ratio * total_income * 0.5

    # 6️⃣ Ratio features (CRITICAL FIX)
    total_outflow_ratio = (
        annual_emi_payments + annual_credit_card_payments
    ) / (total_income + 1)

    liquidity_ratio = (
        interest_from_deposit + dividend_income + mutual_fund_investments
    ) / (total_income + 1)

    # 7️⃣ Risk formula (ratio-driven, scale-independent)
    risk_score = (
        total_outflow_ratio * 4.0
        - liquidity_ratio * 1.5
        + (purchase_of_property > 0) * 0.2
        + np.random.normal(0, 0.2, num_samples)
    )

    risk_prob = 1 / (1 + np.exp(-(risk_score - 0.8) * 4))
    will_fall_in_debt = (
        np.random.random(num_samples) < risk_prob
    ).astype(int)

    df = pd.DataFrame({
        'age': age,
        'salary_income': salary_income,
        'business_receipts': business_receipts,
        'interest_from_deposit': interest_from_deposit,
        'dividend_income': dividend_income,
        'mutual_fund_investments': mutual_fund_investments,
        'sale_of_property': sale_of_property,
        'purchase_of_property': purchase_of_property,
        'annual_credit_card_payments': annual_credit_card_payments,
        'annual_emi_payments': annual_emi_payments,
        'total_outflow_ratio': total_outflow_ratio,  # NEW
        'liquidity_ratio': liquidity_ratio,          # NEW
        'will_fall_in_debt': will_fall_in_debt
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ AIS-scale dataset created")
    print(df['will_fall_in_debt'].value_counts(normalize=True))
    

if __name__ == "__main__":
    synthesize_financial_data()