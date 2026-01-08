# Abstract

## Title
**RE-TabSyn: Controllable Rare-Event Synthetic Data Generation for Financial Tabular Data via Classifier-Free Guidance**

---

## Abstract

Synthetic tabular data generation has emerged as a critical capability for privacy-preserving data sharing and augmentation in financial applications. However, existing methods—including GANs, VAEs, and recent diffusion models—lack mechanisms for controlling class distributions, thereby perpetuating the class imbalance inherent in training data. This limitation is particularly problematic for financial domains where rare events such as fraud, default, or bankruptcy are precisely the classes requiring enhancement.

We present **RE-TabSyn** (Rare-Event Enhanced Tabular Synthesis), a novel framework that combines latent diffusion with Classifier-Free Guidance (CFG) to enable controllable minority class generation. Our approach first encodes mixed-type tabular data into a continuous latent space via a Variational Autoencoder, then performs guided diffusion to generate samples with user-specified class distributions. Through a guidance scale parameter, practitioners can smoothly interpolate between the original data distribution and balanced class ratios.

Extensive experiments on six financial benchmark datasets demonstrate that RE-TabSyn achieves:
- **Competitive fidelity** (KS statistic ~0.15) compared to state-of-the-art methods
- **Controllable minority ratios** boosting minority classes from 11-41% to 47-52%
- **Strong privacy preservation** (DCR > 1.0 across all datasets)
- **Statistical significance** validated across three random seeds

To our knowledge, RE-TabSyn represents the first application of Classifier-Free Guidance to tabular data synthesis, addressing a critical gap in synthetic data research for imbalanced classification tasks.

---

## Keywords

Synthetic Data Generation, Tabular Data, Diffusion Models, Classifier-Free Guidance, Class Imbalance, Financial Data, Privacy-Preserving Machine Learning, Rare Event Detection

---

## ACM Classification

- **Computing methodologies → Machine learning → Generative models**
- **Applied computing → Enterprise computing → Business process management**
- **Security and privacy → Privacy protections**

---

*Word count: 285*
