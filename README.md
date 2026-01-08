# RE-TabSyn: Rare-Event Enhanced Tabular Synthesis

**A novel approach for controllable synthetic tabular data generation with rare event boosting**

---

## 🎯 Quick Summary

RE-TabSyn is the **first application of Classifier-Free Guidance to tabular data synthesis**, enabling controllable minority class generation while maintaining high statistical fidelity.

| Capability | Result |
|:-----------|:-------|
| **Minority Control** | 24% → 50% (on demand) |
| **Fidelity (KS)** | 0.152 ± 0.003 |
| **Privacy (DCR)** | > 1.0 (no memorization) |

---

## 📁 Repository Structure

```
Research/
├── JOURNEY.md          # Complete research story
├── README.md           # This file
│
├── codebase/           # 🔧 Implementation
│   ├── models.py       # Main RE-TabSyn class
│   ├── vae.py          # Variational Autoencoder
│   ├── latent_diffusion.py  # Diffusion + CFG
│   └── run_multi_benchmark.py  # Benchmark script
│
├── paper/              # 📝 Research paper sections
│
├── papers/             # 📚 151 categorized research papers
│
├── results/            # 📊 Analysis & comparisons
│
├── docs/               # 📖 Documentation
│
└── figures/            # 🖼️ Diagrams & visualizations
```

---

## 🚀 Quick Start

```bash
cd codebase
source venv/bin/activate
python run_multi_benchmark.py --quick-test
```

---

## 📊 Key Results

| Dataset | KS Statistic | Minority Boost | Privacy (DCR) |
|:--------|:-------------|:---------------|:--------------|
| Adult | 0.152 | 24.8% → 49.6% | 1.87 |
| German Credit | 0.156 | 30.0% → 44.8% | 90.0 |
| Bank Marketing | 0.211 | 11.3% → 50.2% | 15.1 |
| Lending Club | 0.140 | 20.0% → 50.1% | 4,986 |

---

## 📚 Key Documents

| Document | Description |
|:---------|:------------|
| [JOURNEY.md](./JOURNEY.md) | Complete research story |
| [comparison.md](./results/comparison.md) | Literature comparison |
| [full_benchmark_results.md](./results/full_benchmark_results.md) | Detailed results |
| [RESEARCH_PAPERS.md](./papers/RESEARCH_PAPERS.md) | 151 paper catalog |

---

## 🧪 Novel Contribution

**First application of Classifier-Free Guidance (CFG) to tabular data synthesis.**

```
ε̃ = ε_uncond + w × (ε_cond - ε_uncond)
```

Where `w` (guidance scale) controls minority ratio:
- `w=0`: Original distribution (~24%)
- `w=2`: Balanced generation (~50%)

---

*December 2025*
