# Research Paper Diagrams - Fixed Layout

*All diagrams optimized to prevent arrow overlapping*

---

## Diagram Standards Applied

- **Format:** Mermaid with simplified layouts
- **Layout:** Sequential flow to prevent overlaps
- **Direction:** Consistent TB (top-bottom) or LR (left-right)
- **Subgraphs:** Minimal nesting to avoid rendering issues

---

# 1. High-Level System Pipeline

## Diagram 1.1: RE-TabSyn End-to-End Pipeline

**Caption:** *Complete pipeline from data ingestion to synthetic data generation.*

```mermaid
flowchart LR
    A[Financial Dataset] --> B[Preprocessing]
    B --> C[VAE Training]
    C --> D[Diffusion Training]
    D --> E[CFG Generation]
    E --> F[Synthetic Data]
    F --> G[Evaluation]
```

---

## Diagram 1.2: Detailed Pipeline (Vertical)

**Caption:** *Expanded view of the complete RE-TabSyn pipeline.*

```mermaid
flowchart TB
    A[Financial Dataset] 
    A --> B[Missing Value Imputation]
    B --> C[Categorical Encoding]
    C --> D[Numerical Scaling]
    D --> E[Train/Test Split]
    E --> F[VAE Training - 100 epochs]
    F --> G[Latent Encoding]
    G --> H[Diffusion Training - 100 epochs]
    H --> I[CFG Setup - w=2.0]
    I --> J[Noise Sampling]
    J --> K[Guided Denoising]
    K --> L[VAE Decoding]
    L --> M[Post-Processing]
    M --> N[Synthetic Data - 50% Minority]
```

---

# 2. Model Architecture Diagrams

## Diagram 2.1: VAE Architecture

**Caption:** *Variational Autoencoder structure for tabular data.*

```mermaid
flowchart TB
    I[Input x] --> E1[Linear 256 + ReLU]
    E1 --> E2[Linear 128 + ReLU]
    E2 --> MU[μ Head - Linear 64]
    E2 --> LV[log σ² Head - Linear 64]
    MU --> Z[z = μ + σε]
    LV --> Z
    Z --> D1[Linear 128 + ReLU]
    D1 --> D2[Linear 256 + ReLU]
    D2 --> NUM[Numerical Output]
    D2 --> CAT[Categorical Output]
```

---

## Diagram 2.2: Latent Diffusion Process

**Caption:** *Forward noising and reverse denoising in latent space.*

```mermaid
flowchart LR
    Z0[z₀ Clean] --> Z1[z₁]
    Z1 --> Z2[z₂]
    Z2 --> ZT[zₜ Noise]
    
    ZT --> ZT1[zₜ₋₁]
    ZT1 --> ZT2[zₜ₋₂]
    ZT2 --> Z0G[z₀ Generated]
```

---

## Diagram 2.3: Diffusion Transformer Block

**Caption:** *Single DiT block with AdaLN conditioning.*

```mermaid
flowchart TB
    H[Hidden State h] --> ADALN1[AdaLN Layer 1]
    T[Time Embed] --> ADALN1
    Y[Class Embed] --> ADALN1
    ADALN1 --> ATTN[Multi-Head Attention]
    ATTN --> ADD1[Residual Add]
    H --> ADD1
    ADD1 --> ADALN2[AdaLN Layer 2]
    T --> ADALN2
    Y --> ADALN2
    ADALN2 --> MLP[MLP 256-1024-256]
    MLP --> ADD2[Residual Add]
    ADD1 --> ADD2
    ADD2 --> OUT[Output h']
```

---

## Diagram 2.4: Classifier-Free Guidance

**Caption:** *CFG computation combining conditional and unconditional predictions.*

```mermaid
flowchart TB
    ZT[Noisy Latent zₜ] --> M1[DiT with Class y]
    ZT --> M2[DiT with Null ∅]
    T[Timestep t] --> M1
    T --> M2
    M1 --> COND[εcond]
    M2 --> UNCOND[εuncond]
    COND --> CFG[ε = εuncond + w × εcond - εuncond]
    UNCOND --> CFG
    W[w = 2.0] --> CFG
    CFG --> DENOISE[Denoised zₜ₋₁]
```

---

# 3. Training Pipeline Diagrams

## Diagram 3.1: Two-Phase Training

**Caption:** *Sequential VAE and Diffusion training phases.*

```mermaid
flowchart TB
    subgraph Phase1[Phase 1: VAE]
        P1A[Table x] --> P1B[Encoder]
        P1B --> P1C[Latent z]
        P1C --> P1D[Decoder]
        P1D --> P1E[Loss: MSE + CE + KL]
    end
    
    subgraph Phase2[Phase 2: Diffusion]
        P2A[Latent z₀] --> P2B[Add Noise]
        P2B --> P2C[DiT Predict ε]
        P2C --> P2D[Loss: MSE]
    end
    
    Phase1 --> Phase2
```

---

## Diagram 3.2: Label Dropout for CFG

**Caption:** *10% of labels replaced with null during training.*

```mermaid
flowchart LR
    L1[y=0] --> K1[Keep: y=0]
    L2[y=1] --> K2[Keep: y=1]
    L3[y=0] --> D3[Drop: y=∅]
    L4[y=1] --> K4[Keep: y=1]
    L5[y=0] --> K5[Keep: y=0]
```

---

# 4. Generation Pipeline

## Diagram 4.1: Complete Generation Flow

**Caption:** *Full synthetic data generation from noise to table.*

```mermaid
flowchart TB
    N[Sample zₜ from N 0 I] --> L1[Set target class y=1]
    L1 --> L2[Set guidance w=2.0]
    L2 --> L3[Reverse diffusion T steps]
    L3 --> L4[CFG: blend cond/uncond]
    L4 --> L5[Get clean z₀]
    L5 --> L6[VAE Decode]
    L6 --> L7[Inverse Scale]
    L7 --> L8[Argmax Categories]
    L8 --> L9[Synthetic Table]
```

---

# 5. Evaluation Framework

## Diagram 5.1: Three-Pillar Evaluation

**Caption:** *Fidelity, Utility, and Privacy evaluation metrics.*

```mermaid
flowchart TB
    ROOT[Synthetic Data Quality]
    ROOT --> F[Statistical Fidelity]
    ROOT --> U[ML Utility]
    ROOT --> P[Privacy]
    
    F --> F1[KS Statistic]
    F --> F2[Chi-Square]
    F --> F3[JSD]
    
    U --> U1[TSTR AUC]
    U --> U2[TSTR F1]
    
    P --> P1[DCR]
    P --> P2[MIA Resistance]
```

---

## Diagram 5.2: TSTR Evaluation Protocol

**Caption:** *Train on Synthetic, Test on Real workflow.*

```mermaid
flowchart LR
    S[Synthetic Data] --> T1[Train Classifier]
    T1 --> R[Test on Real Data]
    R --> M1[AUC]
    R --> M2[F1]
    R --> M3[Accuracy]
```

---

# 6. Data Preprocessing

## Diagram 6.1: Preprocessing Pipeline

**Caption:** *Data preparation workflow from raw to model-ready.*

```mermaid
flowchart TB
    RAW[Raw CSV] --> NUM[Numerical Columns]
    RAW --> CAT[Categorical Columns]
    RAW --> TGT[Target Column]
    
    NUM --> N1[Median Impute]
    N1 --> N2[QuantileTransform]
    
    CAT --> C1[Mode Impute]
    C1 --> C2[LabelEncode]
    
    N2 --> CONCAT[Concatenate Features]
    C2 --> CONCAT
    
    CONCAT --> SPLIT[Stratified Split]
    TGT --> SPLIT
    
    SPLIT --> TRAIN[Train Set 80%]
    SPLIT --> TEST[Test Set 20%]
```

---

# 7. Model Comparison

## Diagram 7.1: Capability Matrix

**Caption:** *Comparison of model capabilities.*

```mermaid
flowchart TB
    subgraph GAN[GAN Models]
        G1[CTGAN]
        G2[TVAE]
    end
    
    subgraph DIFF[Diffusion Models]
        D1[TabDDPM - Failed]
        D2[TabSyn]
        D3[RE-TabSyn]
    end
    
    G1 --> CAP1[Moderate Fidelity]
    G2 --> CAP2[Stable Training]
    D1 --> CAP3[Does Not Work]
    D2 --> CAP4[Best Fidelity]
    D3 --> CAP5[Minority Control ✓]
```

---

## Diagram 7.2: Trade-off Comparison

**Caption:** *Performance trade-off between models.*

```mermaid
flowchart LR
    subgraph Fidelity[High Fidelity]
        F1[TabSyn: KS=0.11]
    end
    
    subgraph Balanced[Good Balance]
        B1[CTGAN: KS=0.15]
        B2[RE-TabSyn: KS=0.17]
    end
    
    subgraph Failed[Failed]
        X1[TabDDPM: KS=0.77]
    end
    
    subgraph Control[Has Control]
        C1[RE-TabSyn: 50% minority ✓]
    end
```

---

# 8. Research Methodology Flow

## Diagram 8.1: Research Process

**Caption:** *From problem identification to validation.*

```mermaid
flowchart TB
    P1[TabDDPM Fails - KS 0.80] --> P2[Identify: Discrete diffusion issue]
    P2 --> P3[TabSyn works but no control]
    P3 --> S1[Solution: Latent Space Diffusion]
    S1 --> S2[Add CFG for control]
    S2 --> S3[Transformer backbone]
    S3 --> I1[Implement: VAE 100 epochs]
    I1 --> I2[Implement: Diffusion 100 epochs]
    I2 --> I3[CFG: w=2.0]
    I3 --> V1[Validate: 6 datasets]
    V1 --> V2[Validate: 3 seeds]
    V2 --> V3[Compare: 4 baselines]
```

---

# 9. VAE Training Loop

## Diagram 9.1: VAE Forward Pass

**Caption:** *Single training iteration for VAE.*

```mermaid
flowchart LR
    X[Batch x] --> ENC[Encoder]
    ENC --> MU[μ]
    ENC --> SIG[σ]
    MU --> REP[z = μ + σε]
    SIG --> REP
    REP --> DEC[Decoder]
    DEC --> XHAT[x̂]
    XHAT --> LOSS[L = Recon + KL]
    LOSS --> BACK[Backprop]
```

---

# 10. Diffusion Training Loop

## Diagram 10.1: Diffusion Forward Pass

**Caption:** *Single training iteration for diffusion.*

```mermaid
flowchart LR
    Z0[Latent z₀] --> NOISE[Add noise → zₜ]
    NOISE --> DIT[DiT Model]
    T[Sample t] --> DIT
    Y[Class y or ∅] --> DIT
    DIT --> EPRED[Predicted ε]
    EPRED --> LOSS[MSE Loss]
    LOSS --> BACK[Backprop]
```

---

# 11. CFG Sampling Loop

## Diagram 11.1: Single Denoising Step

**Caption:** *One step of CFG-guided reverse diffusion.*

```mermaid
flowchart TB
    ZT[Current zₜ] --> COND_PASS[DiT with y=1]
    ZT --> UNCOND_PASS[DiT with y=∅]
    COND_PASS --> EC[εcond]
    UNCOND_PASS --> EU[εuncond]
    EC --> BLEND[ε = EU + 2.0 × EC - EU]
    EU --> BLEND
    BLEND --> STEP[zₜ₋₁ = denoise zₜ using ε]
```

---

# 12. Complete System Architecture

## Diagram 12.1: System Block Diagram

**Caption:** *All components of RE-TabSyn system.*

```mermaid
flowchart TB
    subgraph INPUT[Input Layer]
        I1[CSV Data]
    end
    
    subgraph PREP[Preprocessing]
        P1[Impute] --> P2[Encode] --> P3[Scale]
    end
    
    subgraph VAE[VAE Module]
        V1[Encoder] --> V2[Latent z] --> V3[Decoder]
    end
    
    subgraph DIFF[Diffusion Module]
        D1[Forward Process] --> D2[DiT Network] --> D3[Reverse Process]
    end
    
    subgraph CFG[CFG Module]
        C1[Conditional] --> C2[Blend] --> C3[Guided Output]
        C4[Unconditional] --> C2
    end
    
    subgraph EVAL[Evaluation]
        E1[KS] --> E4[Report]
        E2[AUC] --> E4
        E3[DCR] --> E4
    end
    
    INPUT --> PREP
    PREP --> VAE
    VAE --> DIFF
    DIFF --> CFG
    CFG --> EVAL
```

---

# 13. Minority Ratio Control

## Diagram 13.1: Guidance Scale Effect

**Caption:** *How guidance scale affects minority ratio.*

```mermaid
flowchart LR
    W0[w=0.0] --> R0[24.8% minority]
    W1[w=1.0] --> R1[38.5% minority]
    W2[w=2.0] --> R2[49.6% minority ✓]
    W3[w=3.0] --> R3[58.2% minority]
```

---

# 14. Privacy Evaluation

## Diagram 14.1: DCR Computation

**Caption:** *Distance to Closest Record calculation.*

```mermaid
flowchart TB
    SYN[Each synthetic record] --> DIST[Compute distance to all real records]
    DIST --> MIN[Find minimum distance]
    MIN --> DCR[DCR = min distance]
    DCR --> CHECK{DCR > 1.0?}
    CHECK --> |Yes| SAFE[Privacy OK ✓]
    CHECK --> |No| RISK[Privacy Risk ⚠]
```

---

# 15. TSTR vs TRTR Comparison

## Diagram 15.1: Utility Evaluation Types

**Caption:** *Different training/testing combinations.*

```mermaid
flowchart TB
    subgraph TRTR[TRTR - Baseline]
        T1[Train: Real] --> T2[Test: Real]
    end
    
    subgraph TSTR[TSTR - Primary]
        S1[Train: Synthetic] --> S2[Test: Real]
    end
    
    subgraph COMPARE[Compare]
        T2 --> C[Utility Ratio = TSTR/TRTR]
        S2 --> C
    end
```

---

# 16. Dataset Characteristics

## Diagram 16.1: Dataset Overview

**Caption:** *Benchmark datasets and their properties.*

```mermaid
flowchart TB
    subgraph EXTREME[Extreme Imbalance 5%]
        D1[Polish Bankruptcy: 4.8%]
    end
    
    subgraph SEVERE[Severe Imbalance 10-20%]
        D2[Bank Marketing: 11.3%]
        D3[Lending Club: 20.0%]
    end
    
    subgraph MODERATE[Moderate Imbalance 20-35%]
        D4[Adult Income: 24.8%]
        D5[German Credit: 30.0%]
    end
    
    subgraph BALANCED[Near Balanced 40%+]
        D6[Credit Approval: 44.5%]
    end
```

---

# 17. Results Summary

## Diagram 17.1: Model Rankings

**Caption:** *Final performance ranking of all models.*

```mermaid
flowchart LR
    subgraph RANK[Performance Ranking]
        R1[1. TabSyn - Best Fidelity]
        R2[2. RE-TabSyn - Best Control]
        R3[3. CTGAN - Moderate]
        R4[4. TVAE - Stable]
        R5[5. TabDDPM - Failed]
    end
    
    R1 --> R2 --> R3 --> R4 --> R5
```

---

# Diagram Index

| # | Name | Type | Section |
|:--|:-----|:-----|:--------|
| 1.1 | Pipeline Simple | flowchart LR | Overview |
| 1.2 | Pipeline Detailed | flowchart TB | Overview |
| 2.1 | VAE Architecture | flowchart TB | Methodology |
| 2.2 | Diffusion Process | flowchart LR | Methodology |
| 2.3 | DiT Block | flowchart TB | Methodology |
| 2.4 | CFG Mechanism | flowchart TB | Methodology |
| 3.1 | Two-Phase Training | flowchart TB | Implementation |
| 3.2 | Label Dropout | flowchart LR | Implementation |
| 4.1 | Generation Flow | flowchart TB | Implementation |
| 5.1 | Evaluation Framework | flowchart TB | Evaluation |
| 5.2 | TSTR Protocol | flowchart LR | Evaluation |
| 6.1 | Preprocessing | flowchart TB | Datasets |
| 7.1 | Capability Matrix | flowchart TB | Results |
| 7.2 | Trade-off | flowchart LR | Results |
| 8.1 | Research Process | flowchart TB | Methodology |
| 9.1 | VAE Loop | flowchart LR | Implementation |
| 10.1 | Diffusion Loop | flowchart LR | Implementation |
| 11.1 | CFG Step | flowchart TB | Implementation |
| 12.1 | System Architecture | flowchart TB | Overview |
| 13.1 | Guidance Effect | flowchart LR | Results |
| 14.1 | DCR Computation | flowchart TB | Evaluation |
| 15.1 | TSTR vs TRTR | flowchart TB | Evaluation |
| 16.1 | Dataset Overview | flowchart TB | Datasets |
| 17.1 | Model Rankings | flowchart LR | Results |

---

*All diagrams use simplified layouts to prevent arrow overlapping*
*Consistent use of flowchart TB or LR for predictable rendering*
*Minimal subgraph nesting to avoid layout conflicts*
