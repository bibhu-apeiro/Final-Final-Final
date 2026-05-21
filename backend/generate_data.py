import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Configuration
n_samples = 2000
output_path = "data/loan_risk_prediction_dataset.csv"

# Generate features with Indian financial scales
data = {
    'Age': np.random.randint(18, 65, n_samples),
    'Income': np.random.choice(
        np.concatenate([
            np.random.randint(15000, 100000, n_samples // 4),      # Low income
            np.random.randint(100000, 500000, n_samples // 4),     # Middle income
            np.random.randint(500000, 1500000, n_samples // 4),    # Upper middle
            np.random.randint(1500000, 2500000, n_samples // 4),   # High income
        ]),
        n_samples
    ),
    'LoanAmount': np.random.choice(
        np.concatenate([
            np.random.randint(50000, 200000, n_samples // 4),      # Small loans
            np.random.randint(200000, 800000, n_samples // 4),     # Medium loans
            np.random.randint(800000, 2000000, n_samples // 4),    # Large loans
            np.random.randint(2000000, 5000000, n_samples // 4),   # Very large loans
        ]),
        n_samples
    ),
    'CreditScore': np.random.choice(
        np.concatenate([
            np.random.randint(300, 500, n_samples // 3),           # Poor credit
            np.random.randint(500, 700, n_samples // 3),           # Fair credit
            np.random.randint(700, 850, n_samples // 3),           # Good credit
        ]),
        n_samples
    ),
    'YearsExperience': np.random.randint(0, 40, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Education': np.random.choice(
        ['High School', 'Bachelors', 'Masters', 'PhD'],
        n_samples,
        p=[0.30, 0.40, 0.20, 0.10]  # Realistic distribution
    ),
    'City': np.random.choice(
        ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad'],
        n_samples
    ),
    'EmploymentType': np.random.choice(
        ['Salaried', 'Self-Employed', 'Unemployed'],
        n_samples,
        p=[0.55, 0.30, 0.15]  # Realistic distribution
    )
}

df = pd.DataFrame(data)

# Heuristic logic for LoanApproved (Target)
# Designed to produce ~40-45% approval rate with clear separation
def determine_approval(row):
    score = 0

    # --- Credit Score (strongest signal) ---
    if row['CreditScore'] >= 750:
        score += 5
    elif row['CreditScore'] >= 650:
        score += 3
    elif row['CreditScore'] >= 550:
        score += 1
    elif row['CreditScore'] >= 450:
        score -= 2
    else:  # Below 450
        score -= 4

    # --- Loan-to-Income Ratio (critical factor) ---
    lti_ratio = row['LoanAmount'] / max(row['Income'], 1)
    if lti_ratio <= 0.5:
        score += 4
    elif lti_ratio <= 1.0:
        score += 2
    elif lti_ratio <= 2.0:
        score += 0
    elif lti_ratio <= 4.0:
        score -= 2
    else:  # Very high LTI
        score -= 5

    # --- Employment Type ---
    if row['EmploymentType'] == 'Salaried':
        score += 2
    elif row['EmploymentType'] == 'Self-Employed':
        score += 0
    else:  # Unemployed
        score -= 4

    # --- Experience ---
    if row['YearsExperience'] >= 10:
        score += 2
    elif row['YearsExperience'] >= 5:
        score += 1
    elif row['YearsExperience'] <= 1:
        score -= 1

    # --- Age Factor ---
    if row['Age'] < 21:
        score -= 1
    elif row['Age'] > 55:
        score -= 1

    # --- Education Bonus (minor) ---
    if row['Education'] in ['Masters', 'PhD']:
        score += 1

    # --- Hard Reject Rules ---
    # Unemployed with large loan -> almost certainly reject
    if row['EmploymentType'] == 'Unemployed' and row['LoanAmount'] > 500000:
        score -= 3

    # Very low credit + high loan -> reject
    if row['CreditScore'] < 450 and lti_ratio > 2.0:
        score -= 3

    # Add controlled randomness
    score += np.random.normal(0, 1.5)

    # Threshold: score > 4 means approved
    return 1 if score > 4 else 0

df['LoanApproved'] = df.apply(determine_approval, axis=1)

# Ensure directory exists
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

# Save to CSV
df.to_csv(output_path, index=False)

approval_rate = df['LoanApproved'].mean()
print(f"Generated synthetic dataset with {n_samples} samples at {output_path}")
print(f"Approval rate: {approval_rate:.2%}")
print(f"Approved: {df['LoanApproved'].sum()}, Rejected: {(1 - df['LoanApproved']).sum():.0f}")
print(f"\nSample data:")
print(df.head(10))
print(f"\nFeature stats:")
print(df.describe())
