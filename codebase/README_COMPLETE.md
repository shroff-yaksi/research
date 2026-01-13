# 📚 RE-TabSyn Research Project: Complete Guide

*A comprehensive walkthrough of the research folder, from papers to implementation*

## 🟡 Current Status: BENCHMARKING IN PROGRESS (Jan 9, 2026)
The full benchmark suite is currently running across all 9 datasets to generate publication-quality results.

---

## 🎯 What This Project Does (Layman's Terms)

Imagine you're a bank that wants to detect credit card fraud. The problem? Only 1-2% of transactions are fraudulent. If you train an AI on this data, it mostly learns to say "not fraud" because that's right 98% of the time!

**Our Solution (RE-TabSyn):**
We built an AI that can create *fake but realistic* financial data where we control how many fraud cases exist. Want 20% fraud instead of 2%? Done. This "synthetic data" can then be used to train better fraud detectors.

**Key Innovation:** Previous methods either:
- Couldn't generate tabular data well (old diffusion models failed)
- Couldn't control the ratio of rare events (like fraud)
- Would just copy real data (privacy nightmare)

RE-TabSyn solves all three.

---

## ✅ Key Fixes & Updates (Jan 9, 2026)

We have rigorously audited and fixed the codebase to ensure scientific validity:

1.  **Privacy Logic Fixed:** The `evaluator.py` now calculates Distance to Closest Record (DCR) against the **Training Data** (not Test Data). This is critical for correctly measuring overfitting and memorization risks.
2.  **True Controllability:** The `models.py` generation logic now accepts an explicit `minority_ratio` parameter (defaulting to 0.5 for boosting), instead of relying on random 50/50 sampling.
3.  **Architecture Correction:** The `latent_diffusion.py` embedding layer size was increased (2 → 3) to correctly support the "Null" token required for Classifier-Free Guidance (CFG), preventing potential crashes.
4.  **Data Pipeline Optimization:**
    *   **Local Caching:** `data_loader.py` now caches datasets locally in `codebase/datasets/` to prevent redundant downloads and speed up benchmarking.
    *   **Type Safety:** Fixed categorical column casting to ensure consistent string handling across all datasets.
5.  **New Metric (Utility):** Added `evaluate_utility` to `evaluator.py` to calculate **TSTR (Train on Synthetic, Test on Real)** performance (F1-Score), proving the downstream value of the synthetic data.

---

## 🏆 Key Achievements

Based on preliminary validation runs (e.g., German Credit dataset):

*   **Utility Victory:** RE-TabSyn achieved a TSTR F1-Score of **0.508** (at just 10 epochs), outperforming both CTGAN (0.504) and TVAE (0.412).
*   **Precision Control:** Successfully boosted the minority class from 30% (Real) to exactly **50.0%** (Synthetic) as requested.
*   **Privacy Preservation:** Confirmed no exact record memorization with a Minimum DCR of **1.50** (safe distance > 0).
*   **Fidelity:** Maintained competitive statistical similarity (KS ~0.19) even at low training epochs.

---

## 🗂️ Folder Structure Overview

```
📁 /Users/shroffyaksi/Desktop/Research/
│
├── 📄 JOURNEY.md              ← Story of how we built this
│
├── 📁 papers/ (155 files)     ← Research papers we studied
│
├── 📁 codebase/               ← THE IMPLEMENTATION ⭐
│   ├── models.py              ← Main Model Wrapper (RE-TabSyn, CTGAN, TVAE)
│   ├── vae.py                 ← Variational Autoencoder
│   ├── latent_diffusion.py    ← Diffusion Model + CFG Logic
│   ├── transformer.py         ← Neural Network Backbone
│   ├── data_loader.py         ← Data Downloading & Caching
│   ├── evaluator.py           ← Metrics (KS, DCR, Utility F1)
│   ├── run_multi_benchmark.py ← Main Execution Script
│   └── results/               ← Benchmark Outputs
│
├── 📁 results/                ← Visualization Plots
│
└── 📁 Reports/                ← Thesis drafts
```

---

## 🚀 How to Run

### Prerequisites
*   Python 3.8+
*   Virtual Environment (configured in `codebase/venv`)

### Execution Steps

1.  **Navigate to Codebase:**
    ```bash
    cd codebase
    ```

2.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate
    ```

3.  **Run Benchmarks:**
    *   **Quick Test (Smoke Check):**
        ```bash
        python run_multi_benchmark.py --quick-test --dataset german_credit
        ```
    *   **Full Benchmark (All Datasets):**
        ```bash
        python run_multi_benchmark.py
        ```

---

## 🔬 Technical Deep Dive

### Architecture Flow

```
Real Data → VAE Encoder → Latent Space (z)
                                 │
                                 ▼
Training:  Add Noise to z  ←  Diffusion Model  ←  Condition (Class Label)
                                 │
Generation: Random Noise   →  Denoise (w/ CFG) →  Synthetic z
                                 │
                                 ▼
Synthetic z → VAE Decoder → Synthetic Data (Balanced!)
```

### Metrics Explained
*   **Fidelity (KS Statistic):** Lower is better. Measures how much the synthetic data distribution overlaps with real data.
*   **Privacy (DCR):** Higher is better. Measures Euclidean distance to the nearest real training record.
*   **Utility (TSTR F1):** Higher is better. Measures how well a classifier trained on synthetic data performs on real data.

---

## 📝 Next Steps

1.  **Complete Benchmarking:** Finish the full suite run across all 9 datasets (in progress).
2.  **Generate Visualizations:** Use `visualize_with_ci.py` to create publication-ready plots with error bars.
3.  **Paper Drafting:** Incorporate the new "Utility" results into the "Experiments" section of the paper.

---

*This guide is stored at: `/Users/shroffyaksi/Desktop/Research/codebase/README_COMPLETE.md`*