# Evaluation Metrics

## 1. Overview

Evaluating synthetic tabular data requires a multi-dimensional assessment framework addressing three fundamental questions:

1. **Fidelity:** Does the synthetic data statistically resemble the real data?
2. **Utility:** Can models trained on synthetic data perform well on real data?
3. **Privacy:** Does the synthetic data protect individual privacy?

We employ a comprehensive suite of metrics spanning statistical similarity, machine learning utility, and privacy preservation. This section provides formal definitions and interpretation guidelines for each metric.

---

## 2. Statistical Similarity Metrics

Statistical similarity metrics quantify distributional closeness between real data $\mathcal{D}_{real}$ and synthetic data $\mathcal{D}_{syn}$.

### 2.1 Kolmogorov-Smirnov (KS) Test

The KS statistic measures the maximum discrepancy between two empirical cumulative distribution functions (ECDFs).

**Definition.** For continuous random variables with ECDFs $F_{real}$ and $F_{syn}$:

$$D_{KS} = \sup_{x} |F_{real}(x) - F_{syn}(x)|$$

where:
- $F_{real}(x) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[x_i^{real} \leq x]$
- $F_{syn}(x) = \frac{1}{m}\sum_{j=1}^{m}\mathbf{1}[x_j^{syn} \leq x]$

**Two-Sample KS Test:**

The null hypothesis $H_0$: Both samples are drawn from the same distribution.

Test statistic: $D_n = \sqrt{\frac{nm}{n+m}} \cdot D_{KS}$

Asymptotic distribution: $\sqrt{n} D_n \xrightarrow{d} K$ (Kolmogorov distribution)

**Aggregation.** For multi-column datasets, we compute per-column KS statistics and report:

$$\bar{D}_{KS} = \frac{1}{d}\sum_{j=1}^{d} D_{KS}^{(j)}$$

**Interpretation:**

| KS Statistic | Quality Assessment |
|:-------------|:-------------------|
| $< 0.05$ | Excellent |
| $0.05 - 0.10$ | Very Good |
| $0.10 - 0.15$ | Good |
| $0.15 - 0.25$ | Acceptable |
| $> 0.25$ | Poor |

**Implementation:**

```python
from scipy.stats import ks_2samp

def compute_ks(real_col, syn_col):
    statistic, p_value = ks_2samp(real_col, syn_col)
    return statistic

avg_ks = np.mean([compute_ks(real[:, j], syn[:, j]) for j in range(d)])
```

---

### 2.2 Chi-Square (χ²) Test

The chi-square test assesses independence between observed and expected categorical frequencies.

**Definition.** For a categorical variable with $K$ categories:

$$\chi^2 = \sum_{k=1}^{K} \frac{(O_k - E_k)^2}{E_k}$$

where:
- $O_k$ = observed frequency in synthetic data for category $k$
- $E_k$ = expected frequency based on real data proportions: $E_k = n_{syn} \cdot \frac{n_k^{real}}{n_{real}}$

**Degrees of Freedom:** $df = K - 1$

**P-value:** $p = P(\chi^2_{df} > \chi^2_{observed})$

**Interpretation:**
- $p > 0.05$: Fail to reject $H_0$ (distributions are similar) ✓
- $p \leq 0.05$: Reject $H_0$ (significant difference) ✗

**Normalized Chi-Square:**

For comparability across variables with different cardinalities:

$$\chi^2_{norm} = \frac{\chi^2}{n_{syn} \cdot (K - 1)}$$

**Implementation:**

```python
from scipy.stats import chisquare

def compute_chi2(real_col, syn_col, categories):
    real_freq = np.array([np.sum(real_col == c) for c in categories])
    syn_freq = np.array([np.sum(syn_col == c) for c in categories])
    
    # Expected frequencies based on real proportions
    expected = (real_freq / real_freq.sum()) * syn_freq.sum()
    
    statistic, p_value = chisquare(syn_freq, expected)
    return statistic, p_value
```

---

### 2.3 Jensen-Shannon Divergence (JSD)

JSD provides a symmetric, bounded measure of distributional similarity.

**Definition.** For probability distributions $P$ and $Q$:

$$JSD(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$ is the mixture distribution and $D_{KL}$ is the Kullback-Leibler divergence:

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}$$

**Properties:**
- Symmetric: $JSD(P \| Q) = JSD(Q \| P)$
- Bounded: $0 \leq JSD \leq 1$ (using log base 2)
- $JSD = 0$ iff $P = Q$

**Interpretation:**

| JSD | Quality |
|:----|:--------|
| $< 0.05$ | Excellent |
| $0.05 - 0.10$ | Good |
| $0.10 - 0.20$ | Acceptable |
| $> 0.20$ | Poor |

**Implementation:**

```python
from scipy.spatial.distance import jensenshannon

def compute_jsd(real_col, syn_col, bins=50):
    # Discretize continuous variables
    hist_real, edges = np.histogram(real_col, bins=bins, density=True)
    hist_syn, _ = np.histogram(syn_col, bins=edges, density=True)
    
    # Normalize to probabilities
    p = hist_real / hist_real.sum()
    q = hist_syn / hist_syn.sum()
    
    return jensenshannon(p, q) ** 2  # Return squared JSD
```

---

### 2.4 Total Variation Distance (TVD)

TVD measures the maximum difference in probability assignments.

**Definition.** For probability distributions $P$ and $Q$:

$$TVD(P, Q) = \frac{1}{2}\sum_{x} |P(x) - Q(x)| = \sup_{A} |P(A) - Q(A)|$$

**Relationship to JSD:**

$$TVD^2 \leq 2 \cdot JSD$$

**Interpretation:**
- $TVD = 0$: Identical distributions
- $TVD = 1$: Disjoint support

---

### 2.5 Correlation Matrix Divergence

Capturing inter-feature relationships is crucial for downstream utility.

**Definition.** Let $\Sigma_{real}$ and $\Sigma_{syn}$ be correlation matrices of real and synthetic data respectively.

**Frobenius Norm Difference:**

$$\Delta_{corr} = \|\Sigma_{real} - \Sigma_{syn}\|_F = \sqrt{\sum_{i,j}(\sigma_{ij}^{real} - \sigma_{ij}^{syn})^2}$$

**Normalized Version:**

$$\Delta_{corr}^{norm} = \frac{\|\Sigma_{real} - \Sigma_{syn}\|_F}{\|\Sigma_{real}\|_F}$$

**Per-Element Maximum:**

$$\Delta_{max} = \max_{i,j} |\sigma_{ij}^{real} - \sigma_{ij}^{syn}|$$

**Interpretation:**

| $\Delta_{corr}^{norm}$ | Quality |
|:-----------------------|:--------|
| $< 0.10$ | Excellent |
| $0.10 - 0.20$ | Good |
| $> 0.20$ | Poor |

**Implementation:**

```python
def correlation_divergence(real, syn):
    corr_real = np.corrcoef(real.T)
    corr_syn = np.corrcoef(syn.T)
    
    # Handle NaN correlations
    corr_real = np.nan_to_num(corr_real)
    corr_syn = np.nan_to_num(corr_syn)
    
    frobenius = np.linalg.norm(corr_real - corr_syn, 'fro')
    normalized = frobenius / np.linalg.norm(corr_real, 'fro')
    max_diff = np.max(np.abs(corr_real - corr_syn))
    
    return {'frobenius': frobenius, 'normalized': normalized, 'max': max_diff}
```

---

### 2.6 Distribution Overlap Score

The overlap score quantifies the shared probability mass between distributions.

**Definition.** For continuous distributions:

$$\text{Overlap}(P, Q) = \int \min(p(x), q(x)) dx$$

**Discrete Approximation:**

$$\text{Overlap} = \sum_{b=1}^{B} \min(P_b, Q_b)$$

where $P_b, Q_b$ are histogram bin probabilities.

**Properties:**
- $\text{Overlap} = 1$: Identical distributions
- $\text{Overlap} = 0$: Disjoint distributions

---

## 3. Machine Learning Utility Metrics

Utility metrics assess whether synthetic data preserves predictive information.

### 3.1 TSTR: Train on Synthetic, Test on Real

The primary utility metric measures classifier performance when trained on synthetic data.

**Protocol:**

1. Train classifier $f_\theta$ on synthetic data: $f_\theta \leftarrow \text{Train}(\mathcal{D}_{syn})$
2. Evaluate on real test set: $\text{Score} = \text{Evaluate}(f_\theta, \mathcal{D}_{real}^{test})$

**Interpretation:** TSTR score close to train-on-real indicates high utility.

**Baseline Comparison:**

$$\text{Utility Ratio} = \frac{\text{TSTR Score}}{\text{TRTR Score}}$$

where TRTR = Train on Real, Test on Real (upper bound).

| Utility Ratio | Assessment |
|:--------------|:-----------|
| $> 0.95$ | Excellent |
| $0.90 - 0.95$ | Very Good |
| $0.85 - 0.90$ | Good |
| $0.80 - 0.85$ | Acceptable |
| $< 0.80$ | Poor |

### 3.2 TRTS: Train on Real, Test on Synthetic

TRTS evaluates how well real-trained models generalize to synthetic data.

**Protocol:**

1. Train classifier on real data: $f_\theta \leftarrow \text{Train}(\mathcal{D}_{real}^{train})$
2. Evaluate on synthetic data: $\text{Score} = \text{Evaluate}(f_\theta, \mathcal{D}_{syn})$

**Use Case:** Validates that synthetic data lies within the real data manifold; low TRTS suggests out-of-distribution synthetic samples.

### 3.3 Classification Metrics

#### 3.3.1 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Limitation:** Misleading for imbalanced datasets.

#### 3.3.2 F1-Score

The harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

where:
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall} = \frac{TP}{TP + FN}$

**Minority F1:** For imbalanced data, we report F1 for the minority class specifically.

#### 3.3.3 ROC-AUC

Area Under the Receiver Operating Characteristic Curve:

$$\text{AUC} = \int_0^1 TPR(FPR^{-1}(t)) \, dt$$

Equivalently, AUC equals the probability that a randomly chosen positive has higher predicted score than a randomly chosen negative:

$$\text{AUC} = P(\hat{y}_{positive} > \hat{y}_{negative})$$

**Interpretation:**

| AUC | Performance |
|:----|:------------|
| $> 0.90$ | Excellent |
| $0.80 - 0.90$ | Good |
| $0.70 - 0.80$ | Fair |
| $< 0.70$ | Poor |

### 3.4 Utility Evaluation Protocol

**Classifiers Used:**

| Classifier | Configuration |
|:-----------|:--------------|
| Logistic Regression | C=1.0, max_iter=1000 |
| Random Forest | n_estimators=100, max_depth=10 |
| XGBoost | n_estimators=100, learning_rate=0.1 |
| MLP | hidden_layers=(100, 50), max_iter=500 |

**Evaluation Matrix:**

| Training Data | Test Data | Metric Name |
|:--------------|:----------|:------------|
| Real | Real | TRTR (Upper Bound) |
| Synthetic | Real | **TSTR** (Primary) |
| Real | Synthetic | TRTS |
| Synthetic | Synthetic | TSTS |

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_tstr(X_syn, y_syn, X_real_test, y_real_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_syn, y_syn)
    
    y_pred = clf.predict(X_real_test)
    y_prob = clf.predict_proba(X_real_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_real_test, y_pred),
        'f1': f1_score(y_real_test, y_pred),
        'auc': roc_auc_score(y_real_test, y_prob)
    }
```

---

## 4. Privacy Metrics

Privacy metrics assess the risk of information leakage from synthetic data.

### 4.1 Distance to Closest Record (DCR)

DCR measures the minimum distance between each synthetic record and all real records.

**Definition.** For synthetic record $\tilde{\mathbf{x}}_i$ and real dataset $\mathcal{D}_{real}$:

$$DCR_i = \min_{j \in [n_{real}]} \|\tilde{\mathbf{x}}_i - \mathbf{x}_j^{real}\|_2$$

**Aggregated Metrics:**

- **Minimum DCR:** $DCR_{min} = \min_i DCR_i$ (worst-case privacy)
- **5th Percentile DCR:** $DCR_{5\%}$ (robust minimum)
- **Average DCR:** $\bar{DCR} = \frac{1}{n_{syn}}\sum_i DCR_i$

**Interpretation:**

| $DCR_{5\%}$ | Privacy Risk |
|:------------|:-------------|
| $> 1.0$ | Low (synthetic records are distant from real) |
| $0.5 - 1.0$ | Moderate |
| $< 0.5$ | High (potential memorization) |

**Normalization:** For comparability across datasets, we normalize by the median inter-record distance:

$$DCR_{norm} = \frac{DCR}{\text{median}_{j \neq k} \|\mathbf{x}_j - \mathbf{x}_k\|}$$

**Implementation:**

```python
from sklearn.neighbors import NearestNeighbors

def compute_dcr(real_data, syn_data):
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(real_data)
    
    distances, _ = nn.kneighbors(syn_data)
    dcr = distances.flatten()
    
    return {
        'min_dcr': np.min(dcr),
        'percentile_5': np.percentile(dcr, 5),
        'avg_dcr': np.mean(dcr),
        'median_dcr': np.median(dcr)
    }
```

---

### 4.2 Membership Inference Attack (MIA) Resistance

MIA evaluates whether an attacker can determine if a specific record was in the training set.

**Attack Model:**

Given:
- Trained generative model $G$
- Target record $\mathbf{x}^*$
- Access to synthetic samples from $G$

Attacker goal: Determine if $\mathbf{x}^* \in \mathcal{D}_{train}$

**Shadow Model Attack (Stadler et al., 2022):**

1. Train shadow generators $G_1, ..., G_k$ on subsets with known membership
2. Train attack classifier on shadow model outputs
3. Apply attack classifier to target model

**Metrics:**

$$\text{MIA Advantage} = |TPR - FPR| = |P(\hat{m}=1|m=1) - P(\hat{m}=1|m=0)|$$

$$\text{MIA AUC} = \text{ROC-AUC of attack classifier}$$

**Interpretation:**

| MIA AUC | Privacy |
|:--------|:--------|
| $\approx 0.50$ | Excellent (random guessing) |
| $0.50 - 0.60$ | Good |
| $0.60 - 0.70$ | Moderate risk |
| $> 0.70$ | High risk |

**Simplified DCR-based MIA:**

For computational efficiency, we approximate MIA vulnerability using DCR:

$$\text{MIA Risk} \approx \frac{|\{i : DCR_i < \tau\}|}{n_{syn}}$$

where $\tau$ is a "too close" threshold (e.g., $\tau = 0.1 \cdot \text{median distance}$).

---

### 4.3 Attribute Inference Resistance

Attribute inference assesses whether sensitive attributes can be inferred from synthetic data.

**Attack Model:**

Given:
- Partial record $\mathbf{x}_{-s}$ (all attributes except sensitive $s$)
- Synthetic dataset $\mathcal{D}_{syn}$

Attacker goal: Infer sensitive attribute $x_s$

**KNN-based Attack:**

1. Find $k$ nearest neighbors in $\mathcal{D}_{syn}$ based on $\mathbf{x}_{-s}$
2. Predict $\hat{x}_s$ as majority vote among neighbors
3. Measure attack accuracy

**Metrics:**

$$\text{Attribute Inference Accuracy} = \frac{1}{n_{test}}\sum_i \mathbf{1}[\hat{x}_s^{(i)} = x_s^{(i)}]$$

$$\text{Inference Gain} = \text{Attack Accuracy} - \text{Baseline Accuracy}$$

where baseline is predicting the majority class.

**Interpretation:**
- Inference Gain $< 5\%$: Low risk
- Inference Gain $5\% - 15\%$: Moderate risk
- Inference Gain $> 15\%$: High risk

---

### 4.4 Exact Match Rate

The simplest privacy metric counts identical records.

**Definition:**

$$\text{Exact Match Rate} = \frac{|\{\tilde{\mathbf{x}} \in \mathcal{D}_{syn} : \tilde{\mathbf{x}} \in \mathcal{D}_{real}\}|}{|\mathcal{D}_{syn}|}$$

**Relaxed Match (ε-Match):**

$$\text{ε-Match Rate} = \frac{|\{\tilde{\mathbf{x}} : \exists \mathbf{x} \in \mathcal{D}_{real}, \|\tilde{\mathbf{x}} - \mathbf{x}\| < \epsilon\}|}{|\mathcal{D}_{syn}|}$$

**Interpretation:**
- Exact Match $= 0\%$: Required minimum
- Exact Match $> 0\%$: Indicates memorization

---

## 5. Minority Class Preservation Metrics

Specific to our rare-event focus, we evaluate minority class representation.

### 5.1 Minority Ratio

$$\pi_{real} = \frac{|\{y_i = 1\}|}{N}, \quad \pi_{syn} = \frac{|\{\tilde{y}_i = 1\}|}{M}$$

**Ratio Difference:**

$$\Delta\pi = |\pi_{syn} - \pi_{target}|$$

where $\pi_{target}$ is the desired minority ratio (e.g., 0.5 for balanced).

**Interpretation:**
- $\Delta\pi < 0.05$: Excellent control
- $\Delta\pi < 0.10$: Good control
- $\Delta\pi > 0.10$: Poor control

### 5.2 Minority Class Fidelity

KS statistic computed only on minority class samples:

$$D_{KS}^{minority} = \sup_x |F_{real}^{y=1}(x) - F_{syn}^{y=1}(x)|$$

This ensures minority samples are not just present but realistic.

---

## 6. Benchmark Comparison Framework

### 6.1 Evaluation Protocol

For fair comparison, we standardize:

1. **Train/Test Split:** 80/20 stratified
2. **Synthetic Size:** $|\mathcal{D}_{syn}| = |\mathcal{D}_{test}|$
3. **Random Seeds:** 3 seeds (42, 123, 456)
4. **Reporting:** Mean ± Standard Deviation

### 6.2 Benchmark Results Table

| Model | KS ↓ | JSD ↓ | Δ Corr ↓ | TSTR AUC ↑ | F1 (Min) ↑ | DCR ↑ | Minority Ratio |
|:------|:-----|:------|:---------|:-----------|:-----------|:------|:---------------|
| **Real Data** | 0.00 | 0.00 | 0.00 | 0.87 | 0.55 | - | 24.8% |
| CTGAN | 0.15 | 0.12 | 0.18 | 0.82 | 0.48 | 1.2 | 23.1% |
| TVAE | 0.16 | 0.14 | 0.15 | 0.81 | 0.46 | 1.4 | 22.5% |
| TabDDPM | 0.80 | 0.45 | 0.52 | 0.55 | 0.00 | 0.8 | 0.0% |
| TabSyn | **0.10** | **0.08** | **0.10** | **0.85** | 0.52 | **1.8** | 24.2% |
| **RE-TabSyn** | 0.15 | 0.12 | 0.14 | 0.80 | **0.55** | 1.5 | **49.6%** |

### 6.3 Aggregated Comparison (6 Datasets × 3 Seeds)

| Model | Avg KS | Avg AUC | Avg DCR | Minority Control |
|:------|:-------|:--------|:--------|:-----------------|
| CTGAN | 0.16 ± 0.04 | 0.81 ± 0.05 | 1.3 ± 0.4 | ❌ |
| TVAE | 0.17 ± 0.05 | 0.80 ± 0.06 | 1.5 ± 0.5 | ❌ |
| TabDDPM | 0.75 ± 0.12 | 0.58 ± 0.10 | 0.9 ± 0.3 | ❌ |
| TabSyn | **0.11 ± 0.02** | **0.84 ± 0.04** | **1.9 ± 0.6** | ❌ |
| **RE-TabSyn** | 0.17 ± 0.04 | 0.80 ± 0.05 | 1.6 ± 0.8 | **✓ (50%)** |

---

## 7. Summary of Metrics

| Category | Metric | Measures | Target |
|:---------|:-------|:---------|:-------|
| **Fidelity** | KS Statistic | Column distributions | < 0.15 |
| | JSD | Overall similarity | < 0.10 |
| | Correlation Diff | Feature relationships | < 0.15 |
| | Chi-square p-value | Categorical match | > 0.05 |
| **Utility** | TSTR AUC | Predictive usefulness | > 0.85 |
| | TSTR F1 | Minority detection | > 0.50 |
| | Utility Ratio | vs Real baseline | > 0.90 |
| **Privacy** | Min DCR | Closest distance | > 0.5 |
| | 5th %ile DCR | Robust minimum | > 1.0 |
| | MIA AUC | Attack resistance | < 0.60 |
| | Exact Match | Memorization | = 0% |
| **Control** | Minority Ratio Diff | Class control | < 0.05 |

---

*Section word count: ~2,400*
