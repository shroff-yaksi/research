# Research Paper Diagram Codes (Mermaid - Aesthetic Version)

These refined diagrams use professional styling, consistent color coding, and compact layouts for conference paper readiness.

## 1. System Architecture (fig:system_arch)
```mermaid
graph LR
    subgraph Data[Data Layer]
        D1[Raw CSV] 
    end
    
    subgraph Core[RE-TabSyn Core]
        style Core fill:#f0f7ff,stroke:#0056b3
        direction TB
        P[Preprocessing] --> V[Tabular VAE]
        V --> L[Latent Diffusion]
    end
    
    subgraph Cont[Control]
        style Cont fill:#fff9e6,stroke:#d4a017
        CFG[CFG Module\nw=2.0]
    end
    
    subgraph Out[Final Output]
        S[Synthetic Data]
    end
    
    Data --> P
    L -.-> CFG
    CFG --> S
    S --> Eval[Quality Metrics]

    classDef default font-family:Arial,font-size:12px;
```

## 2. VAE Architecture (fig:vae_arch)
```mermaid
flowchart TB
    style LAT fill:#fff9e6,stroke:#d4a017,stroke-width:2px
    
    Input([Input Record x])
    
    subgraph ENC[Encoder]
        style ENC fill:#e3f2fd,stroke:#1e88e5
        E1[Linear 256 + ReLU]
        E2[Linear 128 + ReLU]
        E1 --> E2
    end
    
    subgraph LAT[Latent Space]
        MU[Mean μ]
        LV[Log-Var log σ²]
        MU --> Z((Sampling z))
        LV --> Z
    end
    
    subgraph DEC[Decoder]
        style DEC fill:#e8f5e9,stroke:#43a047
        D1[Linear 128 + ReLU]
        D2[Linear 256 + ReLU]
        D1 --> D2
    end
    
    Output([Reconstruction x̂])

    Input --> E1
    E2 --> MU
    E2 --> LV
    Z --> D1
    D2 --> Output
```

## 3. CFG Mechanism (fig:cfg)
```mermaid
flowchart LR
    style GEN fill:#fff3e0,stroke:#fb8c00,stroke-width:2px
    
    Zt[Noisy Latent zₜ]
    Time([Timestep t])
    Class([Class Label y])
    
    subgraph GEN[CFG Generation Engine]
        direction TB
        M1[DiT Model y]
        M2[DiT Model ∅]
        
        M1 --> E_cond[ε_cond]
        M2 --> E_uncond[ε_uncond]
        
        E_uncond --> Blend{Extrapolate}
        E_cond --> Blend
        W[[Guidance Scale w]] --> Blend
    end
    
    Zt --> M1 & M2
    Time --> M1 & M2
    Class --> M1
    
    Blend --> Zt_prev([Denoised zₜ₋₁])
```

## 4. Evaluation Framework (fig:evaluation)
```mermaid
graph TD
    classDef main fill:#333,color:#fff,stroke-width:2px;
    classDef sub fill:#f5f5f5,stroke:#333;
    
    Quality((Synthetic Quality)):::main
    
    Quality --> Fidelity[Fidelity\nKS-Stat/JSD]:::sub
    Quality --> Utility[ML Utility\nTSTR F1-Score]:::sub
    Quality --> Privacy[Privacy\nDCR Metric]:::sub
    
    Fidelity --- Fdesc[Statistical Accuracy]
    Utility --- Udesc[Predictive Power]
    Privacy --- Pdesc[Copy Detection]
```

## 5. Trade-off Comparison (fig:tradeoff)
```mermaid
flowchart LR
    style TOP fill:#e8f5e9,stroke:#2e7d32
    style MID fill:#fff9e6,stroke:#fbc02d
    style BOT fill:#ffebee,stroke:#c62828
    
    TOP[SOTA Fidelity\nTabSyn: 0.11] 
    MID[Balanced Choice\nRE-TabSyn: 0.17] 
    BOT[Baseline Fail\nTabDDPM: 0.77]
    
    TOP --- MID --- BOT
    
    Control{{Controllability}} -.-> MID
```
