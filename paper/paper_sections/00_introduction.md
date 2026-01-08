# Introduction

## 1. Synthetic Data: Definition and Importance

Synthetic data refers to artificially generated information that statistically mimics real-world datasets without directly copying individual records. Unlike anonymization or pseudonymization—which transform existing data—synthetic data generation creates entirely new samples from learned distributions, offering a fundamentally different approach to data privacy and availability challenges.

The importance of synthetic data has grown exponentially in recent years, driven by three converging forces:

**Privacy Regulations.** Legislation such as the General Data Protection Regulation (GDPR), California Consumer Privacy Act (CCPA), and sector-specific mandates impose strict constraints on personal data usage. Synthetic data provides a pathway to unlock analytical value while maintaining regulatory compliance.

**Data Scarcity.** Many domains suffer from insufficient data for machine learning applications, particularly for rare but critical events. Synthetic data augmentation addresses this fundamental limitation by generating additional samples that preserve distributional properties.

**Collaborative Analytics.** Organizations increasingly seek to share insights across institutional boundaries—between banks, hospitals, or research institutions—without exposing sensitive information. Synthetic data enables collaborative model development without data transfer.

Recent advances in deep generative models—Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and diffusion models—have dramatically improved synthetic data quality, achieving near-indistinguishable fidelity on image and text domains. However, the application to structured tabular data, particularly in high-stakes domains like finance, remains comparatively underexplored and presents unique challenges that motivate this work.

---

## 2. Motivation: Financial Sector Applications

The financial services industry represents an ideal domain for synthetic data research due to its unique confluence of data sensitivity, regulatory pressure, and analytical requirements.

**Sensitivity and Liability.** Financial records contain highly personal information—income, debt levels, transaction histories, creditworthiness assessments—whose exposure carries significant legal and reputational consequences. Unlike research datasets with informed consent, production financial data carries fiduciary obligations that preclude casual sharing or publication.

**Regulatory Constraints.** Financial institutions operate under overlapping regulatory frameworks:
- **GLBA (Gramm-Leach-Bliley Act):** Mandates protection of consumers' nonpublic personal information
- **PCI-DSS:** Governs payment card data handling
- **Basel III/IV:** Requires model validation on representative data
- **Fair Lending Laws:** Necessitate bias auditing across demographic groups

These regulations simultaneously demand rigorous model validation while restricting the data available for such validation—a tension synthetic data can resolve.

**Critical Analytical Needs.** Financial applications include:
- **Fraud Detection:** Identifying rare malicious transactions (0.1–2% prevalence)
- **Credit Risk Modeling:** Predicting default events (5–30% prevalence)
- **Anti-Money Laundering:** Detecting suspicious activity patterns
- **Algorithmic Trading:** Backtesting strategies on diverse market scenarios

Each application shares a common challenge: the events of greatest interest—fraud, default, market crashes—are precisely the rarest in historical data. Models trained on imbalanced datasets systematically underpredict these critical minority events.

---

## 3. Why Tabular Data?

We focus exclusively on tabular (structured) data for several compelling reasons:

**Ubiquity in Enterprise ML.** Despite advances in deep learning for unstructured data, tabular formats dominate enterprise machine learning. Customer records, transaction logs, loan applications, and risk assessments are inherently structured. Gartner estimates that over 80% of enterprise data resides in relational databases and spreadsheets.

**Heterogeneous Feature Types.** Tabular data uniquely combines:
- **Numerical features:** Continuous (income, age) and discrete (count data)
- **Categorical features:** Nominal (occupation, state) and ordinal (education level)
- **Missing values:** Systematically or randomly absent entries
- **Complex constraints:** Business rules, validity ranges, inter-column dependencies

This heterogeneity distinguishes tabular synthesis from image or text generation, where data types are homogeneous.

**Underexplored Research Area.** Image synthesis has achieved remarkable results (StyleGAN, DALL-E, Stable Diffusion), as has text generation (GPT, LLaMA). Tabular synthesis lags significantly behind. Direct application of image-domain techniques fails on structured data, as we demonstrate empirically: TabDDPM, which applies diffusion directly to one-hot encoded tables, achieves KS statistics exceeding 0.80—indicating near-complete distributional mismatch.

**High-Stakes Decisions.** Tabular data underlies consequential decisions affecting individuals: loan approvals, insurance pricing, hiring recommendations, medical diagnoses. The quality of synthetic data directly impacts the fairness and accuracy of these decisions.

---

## 4. Problem Statement and Research Gaps

Despite significant progress in synthetic tabular data generation, a critical limitation persists: **existing methods cannot control the class distribution of generated samples**.

### 4.1 The Class Imbalance Problem

Financial datasets exhibit severe class imbalance:

| Application | Minority Class | Typical Prevalence |
|:------------|:---------------|:-------------------|
| Credit Card Fraud | Fraudulent | 0.1–0.5% |
| Loan Default | Defaulted | 5–15% |
| Company Bankruptcy | Bankrupt | 3–8% |
| Suspicious Transactions | Suspicious | 0.01–1% |

Machine learning models trained on such data learn to predict the majority class, achieving high accuracy while failing catastrophically on minority events—precisely the events of greatest business importance.

### 4.2 Limitations of Existing Approaches

**Traditional Resampling:**
- SMOTE and variants create interpolated samples that may lie outside the true data manifold
- Undersampling discards valuable majority class information
- Neither approach generates truly novel samples

**GAN-based Methods (CTGAN, TVAE):**
- Suffer from mode collapse, "forgetting" minority classes
- Replicate training class distribution without enhancement
- No mechanism for post-hoc ratio adjustment

**Diffusion-based Methods (TabDDPM, TabSyn):**
- TabDDPM fails on mixed-type data (KS > 0.80)
- TabSyn achieves excellent fidelity but mirrors training imbalance
- No support for controllable generation

### 4.3 The Control Gap

We identify a fundamental gap in the literature:

> **No existing synthetic tabular data method enables practitioners to specify the desired class distribution during generation.**

This gap is particularly acute for financial applications, where balanced training data could significantly improve fraud detection, default prediction, and risk assessment.

---

## 5. Contributions

This paper makes the following contributions:

**1. Novel Application of Classifier-Free Guidance to Tabular Data**

We introduce RE-TabSyn (Rare-Event Enhanced Tabular Synthesis), the first application of Classifier-Free Guidance (CFG) to tabular data synthesis. CFG, previously successful in image generation, enables controllable generation without auxiliary classifiers by jointly training conditional and unconditional models.

**2. Controllable Minority Class Generation**

RE-TabSyn allows practitioners to specify target minority ratios via a guidance scale parameter. In our experiments, we boost minority representation from 11–41% (original) to 47–52% (generated), achieving near-balanced datasets on demand.

**3. Improved Minority Class Detection**

We demonstrate that classifiers trained on RE-TabSyn synthetic data achieve **3.1% higher minority F1-score** than those trained on real imbalanced data—the first evidence that synthetic data can outperform real data for minority detection tasks.

**4. Comprehensive Financial Benchmark**

We evaluate RE-TabSyn on six financial datasets spanning credit risk, income prediction, and marketing response. We provide systematic comparison against four baselines (CTGAN, TVAE, TabDDPM, TabSyn) across fidelity, utility, and privacy metrics.

**5. Open-Source Implementation**

We release our implementation, enabling reproducibility and extension of this work.

---

## 6. Scope and Significance

### 6.1 Scope

This work addresses:
- **Binary classification** tabular datasets with class imbalance
- **Single-table** synthesis (relational/multi-table data excluded)
- **Moderate dimensionality** (tested up to 64 features)
- **Financial domain** focus with broader applicability

We do not address:
- Time-series or sequential data
- Multi-class imbalance (extension straightforward)
- Extreme high-dimensional settings (d > 100)

### 6.2 Significance

**Practical Impact.** RE-TabSyn enables financial institutions to:
- Train more effective fraud detection models
- Conduct bias auditing with balanced synthetic data
- Share data across organizational boundaries safely
- Augment rare event datasets for improved modeling

**Research Impact.** This work:
- Bridges controllable image generation techniques to tabular domain
- Establishes benchmarks for minority-aware synthetic evaluation
- Opens research directions in guided tabular synthesis

**Societal Impact.** Improved minority class detection directly benefits:
- Consumers (better fraud protection)
- Financial institutions (reduced losses)
- Regulators (enhanced oversight capability)

---

## 7. Paper Structure

The remainder of this paper is organized as follows:

**Section 2: Literature Review** surveys related work in synthetic data generation, tabular modeling, GANs, VAEs, diffusion models, and privacy-preserving techniques. We systematically analyze 18 peer-reviewed publications and identify the research gap addressed by our work.

**Section 3: Datasets** describes the six financial benchmark datasets, their characteristics, preprocessing pipeline, and suitability for synthetic generation research.

**Section 4: Methodology** presents the theoretical foundations of RE-TabSyn, including the VAE encoder, latent diffusion model, and Classifier-Free Guidance mechanism. We provide mathematical formulations and algorithmic descriptions.

**Section 5: Implementation** details the experimental setup, hardware environment, hyperparameter configurations, and training procedures.

**Section 6: Evaluation Metrics** defines the comprehensive metric suite spanning statistical similarity, machine learning utility, and privacy preservation.

**Section 7: Results and Analysis** presents empirical findings with statistical significance analysis, model comparisons, and domain-specific insights.

**Section 8: Discussion** interprets results, acknowledges limitations, and suggests future research directions.

**Section 9: Conclusion** summarizes contributions and broader implications.

---

## 8. Summary

Synthetic data generation for financial tabular data presents unique challenges at the intersection of privacy requirements, regulatory constraints, and analytical needs. While existing methods achieve reasonable fidelity, they universally fail to address the critical problem of class imbalance—generating synthetic data that mirrors training set limitations rather than enhancing minority representation.

We propose RE-TabSyn, a novel framework that combines latent diffusion with Classifier-Free Guidance to enable, for the first time, controllable minority class generation in tabular data synthesis. Our experiments demonstrate that RE-TabSyn achieves competitive fidelity while uniquely enabling practitioners to specify desired class distributions—a capability with direct practical value for fraud detection, credit risk modeling, and other imbalanced classification tasks in the financial domain.

---

*Section word count: ~1,800*
