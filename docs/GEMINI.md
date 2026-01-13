# RE-TabSyn: Rare-Event Enhanced Tabular Synthesis

## Project Overview

**RE-TabSyn** is a research project focused on generating controllable synthetic tabular data, specifically addressing the challenge of **rare events** (e.g., fraud detection, loan defaults).

It introduces a novel architecture combining:
1.  **Variational Autoencoder (VAE):** To compress tabular data into a continuous latent space.
2.  **Latent Diffusion Model:** To generate data in the latent space.
3.  **Classifier-Free Guidance (CFG):** To explicitly control the ratio of minority class samples during generation (e.g., boosting fraud cases from 1% to 50%).

### Key Achievements
- **Minority Boosting:** Successfully boosts minority class ratios (e.g., Adult dataset 24% → 50%) on demand.
- **Fidelity:** Maintains high statistical fidelity (KS statistic ~0.15).
- **Privacy:** Ensures no memorization of training data (Distance to Closest Record > 1.0).

---

## 🛠 Building and Running

### Prerequisites
- **Python 3.8+**
- **Virtual Environment:** The project relies on a virtual environment.

### Setup & Execution
The primary code resides in the `codebase/` directory.

1.  **Navigate to the codebase:**
    ```bash
    cd codebase
    ```

2.  **Activate the Virtual Environment:**
    ```bash
    source venv/bin/activate
    # Or if using the root venv: source ../.venv/bin/activate
    ```

3.  **Run Benchmarks:**
    - **Quick Test (Verify installation):**
        ```bash
        python run_multi_benchmark.py --quick-test
        ```
    - **Full Benchmark (All datasets):**
        ```bash
        python run_multi_benchmark.py
        ```
    - **Specific Dataset:**
        ```bash
        python run_multi_benchmark.py --dataset german_credit
        ```

### Key Scripts
- `run_multi_benchmark.py`: Main entry point for training and evaluation.
- `run_benchmark.py`: Legacy/single-run script.
- `visualize_results.py` & `visualize_with_ci.py`: Generate plots from result files.

---

## 📂 Directory Structure & Key Files

### `/codebase/` (Implementation)
- **`models.py`**: The "Conductor". `LatentDiffusionWrapper` class orchestrates the VAE and Diffusion models.
- **`vae.py`**: Tabular VAE implementation (Encoder/Decoder) to handle mixed numerical/categorical data.
- **`latent_diffusion.py`**: The Diffusion core. Implements forward noise process and reverse denoising with CFG.
- **`transformer.py`**: The neural backbone (DiT-style Transformer with Adaptive Layer Norm) for the diffusion model.
- **`data_loader.py`**: Handles downloading and preprocessing of 9 financial datasets (Adult, German Credit, Bank Marketing, etc.).
- **`evaluator.py`**: Computes metrics: KS Statistic (Fidelity), DCR (Privacy), and Minority Ratio.

### `/literature review papers/` (Research Context)
- Contains **150+ categorized research papers** (Diffusion, GANs, Privacy, Transformers).
- Organized by category (A: Diffusion, B: Privacy, C: GANs, etc.).
- **`RESEARCH_PAPERS.md`**: Catalog of all papers.

### `/results/` (Outputs)
- Stores benchmark logs, CSVs of synthetic data, and comparison plots.
- **`comparison.md`**: Comparisons against SOTA (TabDDPM, CTGAN).

### `/docs/` & `/Reports/`
- **`JOURNEY.md`**: The narrative of the research process and decisions.
- **`Reports/`**: Drafts for the thesis/paper.

---

## 💻 Development Conventions

- **Architecture:** PyTorch-based.
- **Data Flow:** `Raw Data` -> `DataLoader` -> `VAE (Compression)` -> `Latent Diffusion (Generation)` -> `VAE (Decompression)` -> `Synthetic Data`.
- **CFG Implementation:**
    - **Training:** Randomly drop class labels (p=0.1) to train unconditional generation.
    - **Inference:** `ε = ε_uncond + w * (ε_cond - ε_uncond)`.
    - **Guidance Scale (`w`):** Controls the strength of the minority class boost. Default `w=2.0`.
- **Metrics:**
    - **Fidelity:** KS Statistic (lower is better), TVD.
    - **Utility:** Machine Learning efficacy (Train on Synthetic, Test on Real).
    - **Privacy:** DCR (Distance to Closest Record).

---

## 📝 TODOs & Next Steps
- Expand benchmarks to more datasets.
- Perform ablation studies on guidance scale `w`.
- Finalize the research paper in `paper/` directory.
