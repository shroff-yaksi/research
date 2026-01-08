# Financial Datasets for RE-TabSyn Benchmarking

This folder documents all **financial tabular datasets** used for benchmarking RE-TabSyn.

---

## Dataset Summary (10 Financial Datasets)

| # | Dataset Name | Rows | Features | Domain | Minority Class | Imbalance | Source |
|:--|:-------------|:-----|:---------|:-------|:---------------|:----------|:-------|
| 1 | Adult (Census Income) | 45,222 | 8 | Income | >50K (24.8%) | ~1:3 | UCI |
| 2 | Credit Card Default | 30,000 | 24 | Credit | Default (22%) | ~1:4 | UCI |
| 3 | German Credit | 1,000 | 21 | Credit | Bad Risk (30%) | ~1:2 | UCI |
| 4 | Bank Marketing | 41,188 | 21 | Banking | Subscribed (11.3%) | ~1:8 | UCI |
| 5 | Australian Credit | 690 | 15 | Credit | Approved (44%) | ~1:1 | UCI |
| 6 | Credit Approval | 690 | 16 | Credit | Approved (44%) | ~1:1 | UCI |
| 7 | Lending Club | 10,000* | 13 | P2P Lending | Default (20%) | ~1:4 | Kaggle |
| 8 | Give Me Some Credit | 10,000* | 11 | Credit | Delinquent (7%) | ~1:13 | Kaggle |
| 9 | Polish Bankruptcy | 5,000* | 11 | Corporate | Bankrupt (5%) | ~1:19 | UCI |

*Synthetic/sampled versions for demonstration

---

## Core Financial Datasets (from UCI)

### 1. Adult (Census Income)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
- **Task**: Predict whether income exceeds $50K/yr
- **Target**: `salary` (>50K / <=50K)
- **Minority**: >50K (24.8%)
- **Features**: age, workclass, education, marital-status, occupation, sex, hours-per-week

### 2. Credit Card Default (Taiwan)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Task**: Predict credit card default next month
- **Target**: `default` (0=no, 1=yes)
- **Minority**: Default (22%)
- **Citation**: Yeh, I. C., & Lien, C. H. (2009)

### 3. German Credit Data
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`
- **Task**: Classify customers as good or bad credit risks
- **Target**: `class` (0=good, 1=bad)
- **Minority**: Bad credit (30%)

### 4. Bank Marketing
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Task**: Predict if client will subscribe to term deposit
- **Target**: `y` (0=no, 1=yes)
- **Minority**: Subscribed (11.3%)
- **Citation**: Moro et al. (2014)

### 5. Australian Credit Approval
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval)
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat`
- **Task**: Credit approval decision
- **Target**: `class` (0=rejected, 1=approved)
- **Minority**: Approved (44%)

### 6. Credit Approval (CRX)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/27/credit+approval)
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data`
- **Task**: Credit card application approval
- **Target**: `class` (+/-)
- **Features**: All anonymized (A1-A15)

---

## Extended Financial Datasets (from Kaggle)

### 7. Lending Club Loan Data
- **Source**: [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Task**: Predict loan default in P2P lending
- **Target**: `loan_status` (0=paid, 1=default)
- **Minority**: Default (20%)
- **Features**: loan_amnt, term, int_rate, grade, emp_length, annual_inc, dti, etc.

### 8. Give Me Some Credit
- **Source**: [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Task**: Predict serious delinquency within 2 years
- **Target**: `SeriousDlqin2yrs` (0=no, 1=yes)
- **Minority**: Delinquent (7%)
- **Features**: RevolvingUtilization, age, DebtRatio, MonthlyIncome, etc.

### 9. Polish Companies Bankruptcy
- **Source**: [UCI](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- **Task**: Predict company bankruptcy
- **Target**: `bankrupt` (0=no, 1=yes)
- **Minority**: Bankrupt (5%)
- **Features**: Financial ratios (net profit/assets, liabilities/assets, EBIT, etc.)

---

## Usage

```python
from data_loader import DataLoader

# Load any financial dataset
loader = DataLoader('german_credit')
loader.load_data()
print(f"Shape: {loader.data.shape}")
print(f"Target: {loader.target_col}")

# Get train/test split
train_data, test_data = loader.split_data()

# Get minority class statistics
stats = loader.get_rare_event_stats()
print(f"Minority ratio: {stats['minority_ratio']:.2%}")
```

### Available Datasets

```python
# Financial datasets (9)
'adult'             # Census Income
'credit_default'    # Taiwan Credit Default
'german_credit'     # German Credit Risk
'bank_marketing'    # Bank Term Deposit
'australian_credit' # Australian Credit Approval
'credit_approval'   # Credit Approval (anonymized)
'lending_club'      # P2P Lending Default
'give_me_credit'    # Credit Delinquency
'polish_bankruptcy' # Company Bankruptcy
```

---

## Why Financial Datasets?

Financial datasets are ideal for evaluating rare event generation because:

1. **Natural Class Imbalance**: Fraud, default, and bankruptcy are inherently rare events
2. **High Stakes**: Accurate minority class detection is critical in finance
3. **Well-Studied**: Extensive literature for comparison
4. **Mixed Data Types**: Combine numerical and categorical features
5. **Privacy Concerns**: Synthetic data generation is highly relevant

---

## Running the Financial Benchmark

```bash
cd /Users/shroffyaksi/Desktop/Research/codebase
source venv/bin/activate

# Quick test (10 epochs, ~45 min)
python run_multi_benchmark.py --quick-test

# Full benchmark (100 epochs, ~6-8 hours)
python run_multi_benchmark.py

# Single dataset
python run_multi_benchmark.py --dataset german_credit
```

---

*Last Updated: 2025-12-09*
