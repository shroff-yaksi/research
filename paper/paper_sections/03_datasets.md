# Datasets

## 1. Overview

This section describes the benchmark datasets employed to evaluate RE-TabSyn and baseline synthetic data generation methods. We curate a diverse collection of six financial and credit-related tabular datasets, selected to represent the heterogeneous nature of real-world financial machine learning applications.

---

## 2. Motivation for Financial Tabular Data

### 2.1 Why Financial Data?

Financial datasets present unique challenges and opportunities for synthetic data research:

**Sensitivity and Regulatory Constraints.** Financial data contains personally identifiable information (PII) subject to stringent regulations including GDPR, CCPA, and sector-specific mandates such as GLBA (Gramm-Leach-Bliley Act) and PCI-DSS. Synthetic data generation offers a privacy-preserving pathway for data sharing, model development, and regulatory compliance testing without exposing real customer information.

**Limited Data Accessibility.** Production financial datasets remain proprietary due to competitive advantage and liability concerns. Public benchmarks enable reproducible research while approximating the statistical properties of real-world financial data. Synthetic generation further democratizes access to realistic financial data distributions.

**High-Stakes Decision Making.** Credit scoring, fraud detection, and loan approval systems directly impact individual financial well-being. Training data quality—particularly regarding minority class representation—critically influences model fairness and performance. Imbalanced datasets lead to biased models that disproportionately affect vulnerable populations.

**Class Imbalance Prevalence.** Financial datasets exhibit severe class imbalance: fraud rates typically range from 0.1% to 2%, default rates from 5% to 30%, and bankruptcy incidence below 5%. Traditional approaches either oversample minorities (risking overfitting) or undersample majorities (discarding valuable information). Controllable synthetic generation addresses this fundamental challenge.

### 2.2 Why Tabular Data?

We focus exclusively on tabular (structured) data for the following reasons:

**Heterogeneous Feature Types.** Tabular data combines numerical features (income, age, transaction amount), categorical features (occupation, marital status, loan purpose), and ordinal features (credit rating, education level). This heterogeneity poses unique challenges for generative models designed for homogeneous data types (e.g., images, text).

**Non-Euclidean Feature Space.** Unlike images where pixels occupy a continuous 2D grid, tabular features lack spatial or sequential relationships. Each column represents an independent semantic dimension with distinct statistical properties, distributions, and valid ranges.

**Real-World ML Dependency.** Tabular data remains the dominant format in enterprise machine learning. Despite advances in deep learning for unstructured data, gradient boosted trees and logistic regression on tabular features power the majority of production ML systems in finance, healthcare, and e-commerce.

**Underexplored in Generative AI.** While image and text generation have received extensive attention (GANs, diffusion models, LLMs), tabular synthesis remains comparatively understudied. Recent work (TabSyn, TabDDPM) demonstrates that naive applications of image-domain techniques fail on structured data, motivating specialized approaches.

---

## 3. Dataset Descriptions

We evaluate on six publicly available datasets spanning credit risk, income prediction, and marketing response—all relevant to financial services applications.

### 3.1 Adult Income (Census)

| Property | Value |
|:---------|:------|
| **Source** | UCI Machine Learning Repository |
| **Task** | Binary classification (income ≤50K vs >50K) |
| **Rows** | 45,222 (after cleaning) |
| **Columns** | 14 (reduced to 8 key features) |
| **Numerical** | 2 (age, hours-per-week) |
| **Categorical** | 6 (workclass, education, marital-status, occupation, relationship, native-country) |
| **Target** | `salary` (binary) |
| **Minority Class** | >50K income (24.78%) |
| **Missing Values** | 7.4% (occupation, workclass, native-country) |

**Description:** Extracted from 1994 US Census data, the Adult dataset predicts whether annual income exceeds $50,000 based on demographic features. Despite its age, it remains the most widely used tabular benchmark in synthetic data literature, enabling direct comparison with prior work.

**Relevance:** Serves as a proxy for creditworthiness assessment, where income level correlates with loan approval decisions.

---

### 3.2 German Credit

| Property | Value |
|:---------|:------|
| **Source** | UCI Machine Learning Repository |
| **Task** | Binary classification (good vs bad credit risk) |
| **Rows** | 1,000 |
| **Columns** | 20 |
| **Numerical** | 7 (duration, credit_amount, installment_rate, residence, age, existing_credits, dependents) |
| **Categorical** | 13 (account_status, credit_history, purpose, savings, employment, personal_status, guarantors, property, other_plans, housing, job, telephone, foreign_worker) |
| **Target** | `credit_risk` (binary: 1=good, 2=bad) |
| **Minority Class** | Bad credit (30.0%) |
| **Missing Values** | None |

**Description:** Classic credit scoring dataset from a German bank containing applicant attributes and credit outcomes. Small sample size (n=1,000) tests generator performance in low-data regimes.

**Relevance:** Direct credit risk application with human-interpretable features reflecting real underwriting criteria.

---

### 3.3 Bank Marketing

| Property | Value |
|:---------|:------|
| **Source** | UCI Machine Learning Repository |
| **Task** | Binary classification (term deposit subscription) |
| **Rows** | 41,188 |
| **Columns** | 20 |
| **Numerical** | 10 (age, balance, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, nr.employed) |
| **Categorical** | 10 (job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome) |
| **Target** | `y` (yes/no subscription) |
| **Minority Class** | Subscription (11.26%) |
| **Missing Values** | 1.8% (encoded as "unknown") |

**Description:** Portuguese bank direct marketing campaign data for term deposit subscription. Features include client demographics, contact information, and macroeconomic indicators.

**Relevance:** Represents marketing analytics use case with severely imbalanced outcomes and temporal economic features.

---

### 3.4 Credit Approval (Australian)

| Property | Value |
|:---------|:------|
| **Source** | UCI Machine Learning Repository (StatLog) |
| **Task** | Binary classification (credit approval) |
| **Rows** | 690 |
| **Columns** | 15 |
| **Numerical** | 6 (A2, A3, A8, A11, A14, A15) |
| **Categorical** | 9 (A1, A4, A5, A6, A7, A9, A10, A12, A13) |
| **Target** | `A16` (approved/denied) |
| **Minority Class** | Denied (44.5%) |
| **Missing Values** | 5.0% (scattered across features) |

**Description:** Anonymized Australian credit card applications with attribute names replaced for confidentiality. Near-balanced classes enable evaluation of generation quality independent of imbalance effects.

**Relevance:** Tests synthetic generation on privacy-anonymized financial data with hidden feature semantics.

---

### 3.5 Lending Club Loan

| Property | Value |
|:---------|:------|
| **Source** | Kaggle / Lending Club Public Data |
| **Task** | Binary classification (loan default) |
| **Rows** | 10,000 (sampled from full dataset) |
| **Columns** | 12 |
| **Numerical** | 8 (loan_amnt, int_rate, installment, annual_inc, dti, open_acc, revol_bal, total_acc) |
| **Categorical** | 4 (term, grade, home_ownership, purpose) |
| **Target** | `loan_status` (default/paid) |
| **Minority Class** | Default (19.95%) |
| **Missing Values** | 2.3% (employment length, title) |

**Description:** Peer-to-peer lending platform data containing loan characteristics and borrower attributes. Reflects modern fintech credit decisioning with alternative data sources.

**Relevance:** Contemporary financial dataset with realistic feature engineering and credit bureau-derived attributes.

---

### 3.6 Polish Bankruptcy

| Property | Value |
|:---------|:------|
| **Source** | UCI Machine Learning Repository |
| **Task** | Binary classification (company bankruptcy) |
| **Rows** | 5,000 (sampled) |
| **Columns** | 64 (financial ratios) |
| **Numerical** | 64 (all financial ratios) |
| **Categorical** | 0 |
| **Target** | `class` (bankrupt/solvent) |
| **Minority Class** | Bankrupt (4.8%) |
| **Missing Values** | 12.5% (derived ratio undefined) |

**Description:** Polish company financial statements with 64 financial ratios predicting bankruptcy within specified periods. Purely numerical dataset with severe minority imbalance.

**Relevance:** Corporate credit risk scenario with extreme imbalance (4.8% bankruptcy rate) and high dimensionality.

---

## 4. Dataset Characteristics Summary

### 4.1 Aggregate Statistics

| Dataset | N | Features | Num. | Cat. | Minority % | Missing % |
|:--------|:--|:---------|:-----|:-----|:-----------|:----------|
| Adult Income | 45,222 | 8 | 2 | 6 | 24.78% | 7.4% |
| German Credit | 1,000 | 20 | 7 | 13 | 30.0% | 0.0% |
| Bank Marketing | 41,188 | 20 | 10 | 10 | 11.26% | 1.8% |
| Credit Approval | 690 | 15 | 6 | 9 | 44.5% | 5.0% |
| Lending Club | 10,000 | 12 | 8 | 4 | 19.95% | 2.3% |
| Polish Bankruptcy | 5,000 | 64 | 64 | 0 | 4.8% | 12.5% |

### 4.2 Class Imbalance Distribution

```
Polish Bankruptcy  ████░░░░░░░░░░░░░░░░  4.8%
Bank Marketing     ██████░░░░░░░░░░░░░░ 11.3%
Lending Club       ████████░░░░░░░░░░░░ 20.0%
Adult Income       ██████████░░░░░░░░░░ 24.8%
German Credit      ████████████░░░░░░░░ 30.0%
Credit Approval    ██████████████████░░ 44.5%
                   0%                  50%
```

The dataset collection spans a wide range of imbalance ratios (4.8% to 44.5%), enabling evaluation of synthetic generation methods across varying difficulty levels.

### 4.3 Feature Type Distribution

| Dataset | % Numerical | % Categorical | Mixed-Type Complexity |
|:--------|:------------|:--------------|:----------------------|
| Adult Income | 25% | 75% | High (categorical-heavy) |
| German Credit | 35% | 65% | High (many categories) |
| Bank Marketing | 50% | 50% | Medium (balanced) |
| Credit Approval | 40% | 60% | Medium |
| Lending Club | 67% | 33% | Low (numerical-heavy) |
| Polish Bankruptcy | 100% | 0% | Low (pure numerical) |

---

## 5. Preprocessing Pipeline

We apply a standardized preprocessing pipeline to ensure consistent evaluation across datasets and methods.

### 5.1 Missing Value Handling

```python
# Strategy: Mode imputation for categorical, median for numerical
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column].fillna(dataset[column].mode()[0], inplace=True)
    else:
        dataset[column].fillna(dataset[column].median(), inplace=True)
```

**Rationale:** Simple imputation preserves sample size while introducing minimal distributional assumptions. Advanced imputation (MICE, KNN) was avoided to isolate generator performance from imputation quality.

### 5.2 Categorical Encoding

**Strategy: Label Encoding for Low-Cardinality, Target Encoding for High-Cardinality**

| Cardinality | Strategy | Example |
|:------------|:---------|:--------|
| ≤10 unique values | Label Encoding | gender, loan_grade |
| >10 unique values | Target Encoding | occupation, native_country |

```python
from sklearn.preprocessing import LabelEncoder

# Label encoding (invertible for generation)
encoder = LabelEncoder()
encoded = encoder.fit_transform(categorical_column)
```

**Rationale:** One-hot encoding creates sparse, high-dimensional representations problematic for generative models. Label encoding with subsequent VAE processing enables the model to learn semantic relationships between categories.

### 5.3 Numerical Scaling

**Strategy: Quantile Transformation to Gaussian Distribution**

```python
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
scaled = scaler.fit_transform(numerical_columns)
```

**Rationale:** Financial data often exhibits heavy tails, multimodality, and bounded ranges. Quantile transformation:
1. Maps arbitrary distributions to standard Gaussian
2. Enables stable VAE training with Gaussian priors
3. Preserves rank ordering (invertible transformation)

### 5.4 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # Preserve class distribution
    random_state=seed
)
```

| Split | Purpose | Size |
|:------|:--------|:-----|
| Training (80%) | Generator training | Variable |
| Test (20%) | Evaluation holdout | Variable |

**Stratification:** Ensures identical class distributions in train and test sets, critical for imbalanced datasets where random splits may produce unrepresentative partitions.

### 5.5 Preprocessing Pipeline Summary

```
┌─────────────────┐
│   Raw Dataset   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Missing Value   │──► Mode (cat.) / Median (num.)
│   Imputation    │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Categorical   │──► Label Encoding
│    Encoding     │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Numerical     │──► Quantile Transformer (Gaussian)
│    Scaling      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Stratified Split│──► 80% Train / 20% Test
└────────┬────────┘
         ▼
┌─────────────────┐
│ Ready for Model │
└─────────────────┘
```

---

## 6. Ethical Considerations and Risks

### 6.1 Data Provenance

All datasets are publicly available from established repositories (UCI, Kaggle) with appropriate usage licenses. No private or proprietary data was collected for this research.

| Dataset | License | Ethical Clearance |
|:--------|:--------|:------------------|
| Adult Income | CC BY 4.0 | US Census public data |
| German Credit | CC BY 4.0 | Anonymized bank data |
| Bank Marketing | CC BY 4.0 | Published academic data |
| Credit Approval | CC BY 4.0 | Anonymized (A1-A15) |
| Lending Club | Public | Aggregated loan data |
| Polish Bankruptcy | CC BY 4.0 | Public financial filings |

### 6.2 Potential Risks

**Re-identification Risk.** Although datasets are anonymized, synthetic generation on small datasets (German Credit, n=1,000) may produce samples similar to training records. We mitigate this through:
- Distance to Closest Record (DCR) evaluation
- Holdout test set for privacy assessment
- Guidance against releasing synthetic data as real

**Fairness Implications.** Models trained on synthetic data may inherit or amplify biases present in original datasets. The Adult dataset, for instance, reflects 1994 US demographic patterns that may not represent current populations.

**Downstream Misuse.** Synthetic financial data should not be used for:
- Credit decisions affecting real individuals
- Regulatory reporting as substitute for genuine data
- Training production models without validation

### 6.3 Responsible Use Guidelines

1. **Transparency:** Clearly label synthetic data as artificially generated
2. **Validation:** Verify synthetic data utility on downstream tasks before deployment
3. **Privacy Audit:** Conduct DCR and membership inference evaluations
4. **Fairness Testing:** Assess model performance across demographic subgroups

---

## 7. Suitability for Synthetic Generation Research

The curated benchmark suite is particularly suitable for evaluating synthetic tabular data methods due to:

### 7.1 Diversity of Characteristics

- **Sample sizes** ranging from 690 to 45,222 (small to medium scale)
- **Dimensionality** from 8 to 64 features (varying complexity)
- **Feature types** from pure numerical to categorical-dominant (heterogeneity testing)
- **Imbalance ratios** from 4.8% to 44.5% (rare event spectrum)

### 7.2 Established Baselines

All datasets have been used in prior synthetic data literature (CTGAN, TabSyn, benchmarks), enabling direct comparison with published results.

### 7.3 Domain Relevance

Financial applications represent a primary use case for synthetic data—privacy constraints, regulatory requirements, and data scarcity drive practical demand for high-quality financial data synthesis.

### 7.4 Minority Class Emphasis

The consistent presence of minority classes across datasets directly supports evaluation of rare event generation—the core contribution of RE-TabSyn.

---

## 8. Data Availability

All datasets and preprocessing code are publicly available:

- **UCI Repository:** https://archive.ics.uci.edu/
- **Kaggle:** https://www.kaggle.com/datasets
- **Our Code:** [Repository link to be added upon publication]

Preprocessed datasets and data loading utilities are included in the supplementary materials to ensure reproducibility.

---

*Section word count: ~2,100*
