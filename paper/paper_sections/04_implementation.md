# Implementation and Experimental Setup

## 1. Overview

This section details the implementation of RE-TabSyn and baseline methods, experimental configuration, and evaluation protocols. We provide sufficient detail to enable full reproducibility of reported results.

---

## 2. Hardware and Software Environment

### 2.1 Computational Infrastructure

| Component | Specification |
|:----------|:--------------|
| **CPU** | Apple M2 Pro (10-core) / Intel Xeon E5-2690 v4 |
| **GPU** | Apple M2 Pro Neural Engine / NVIDIA Tesla V100 (16GB) |
| **RAM** | 32 GB unified memory |
| **Storage** | 512 GB SSD |
| **OS** | macOS Ventura 13.4 / Ubuntu 20.04 LTS |

### 2.2 Software Stack

| Component | Version | Purpose |
|:----------|:--------|:--------|
| **Python** | 3.10.12 | Primary language |
| **PyTorch** | 2.1.0 | Deep learning framework |
| **NumPy** | 1.24.3 | Numerical computing |
| **Pandas** | 2.0.3 | Data manipulation |
| **Scikit-learn** | 1.3.0 | Preprocessing, evaluation |
| **SDV (CTGAN/TVAE)** | 1.5.0 | Baseline implementations |
| **SciPy** | 1.11.1 | Statistical tests |
| **tqdm** | 4.65.0 | Progress visualization |

### 2.3 Reproducibility

All experiments are seeded for reproducibility:

```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

We report results across three random seeds (42, 123, 456) and provide mean ± standard deviation for all metrics.

---

## 3. End-to-End Pipeline Architecture

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐ │
│  │   Data   │───▶│ Preprocessing│───▶│Model Training │───▶│Generation│ │
│  │ Ingestion│    │   Module     │    │   Pipeline    │    │  Module  │ │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────────┘ │
│       │                │                    │                   │       │
│       ▼                ▼                    ▼                   ▼       │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐ │
│  │  UCI/    │    │• Imputation  │    │• VAE Training │    │• Latent  │ │
│  │  Kaggle  │    │• Encoding    │    │• Diffusion    │    │  Sampling│ │
│  │  Loader  │    │• Scaling     │    │• CFG Setup    │    │• Decoding│ │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      EVALUATION MODULE                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │ Statistical│  │  ML Utility│  │  Privacy   │  │ Benchmarking│  │   │
│  │  │  Metrics   │  │   Metrics  │  │  Metrics   │  │   Module   │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Ingestion Module

The data ingestion module provides unified access to benchmark datasets with automatic download and caching:

```python
class DataLoader:
    """Unified data loading with fallback sources."""
    
    DATASETS = {
        'adult': {'url': 'https://archive.ics.uci.edu/...', 'target': 'salary'},
        'german_credit': {'url': '...', 'target': 'credit_risk'},
        'bank_marketing': {'url': '...', 'target': 'y'},
        'credit_approval': {'url': '...', 'target': 'A16'},
        'lending_club': {'url': '...', 'target': 'loan_status'},
        'polish_bankruptcy': {'url': '...', 'target': 'class'},
    }
    
    def load_data(self, dataset_name):
        """Load dataset with automatic download and preprocessing."""
        config = self.DATASETS[dataset_name]
        data = self._download_or_cache(config['url'])
        return self._preprocess(data, config['target'])
```

**Features:**
- Multi-source fallback (primary URL, mirrors, synthetic fallback)
- Automatic caching to avoid redundant downloads
- Unified interface across heterogeneous sources

### 3.3 Preprocessing Module

Preprocessing transforms raw data into model-ready tensors:

```python
class Preprocessor:
    def __init__(self, categorical_cols, numerical_cols):
        self.cat_encoder = LabelEncoder()
        self.num_scaler = QuantileTransformer(output_distribution='normal')
    
    def fit_transform(self, data):
        """Fit and transform training data."""
        # Categorical encoding
        cat_encoded = self.cat_encoder.fit_transform(data[self.cat_cols])
        
        # Numerical scaling
        num_scaled = self.num_scaler.fit_transform(data[self.num_cols])
        
        return np.concatenate([num_scaled, cat_encoded], axis=1)
    
    def inverse_transform(self, synthetic):
        """Convert synthetic data back to original space."""
        num_original = self.num_scaler.inverse_transform(synthetic[:, :self.n_num])
        cat_original = self.cat_encoder.inverse_transform(synthetic[:, self.n_num:])
        return pd.DataFrame(...)
```

---

## 4. Model Configurations

### 4.1 RE-TabSyn (Proposed Method)

RE-TabSyn comprises three components: VAE encoder/decoder, latent diffusion model, and classifier-free guidance mechanism.

#### 4.1.1 VAE Architecture

| Component | Configuration |
|:----------|:--------------|
| **Encoder** | Linear(d_in, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 2×d_latent) |
| **Latent Dimension** | 64 |
| **Decoder** | Linear(d_latent, 128) → ReLU → Linear(128, 256) → ReLU → Linear(256, d_out) |
| **Output Heads** | Numerical: Linear → identity; Categorical: Linear → Softmax |

```python
class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[256, 128]):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
```

#### 4.1.2 Latent Diffusion Model

| Component | Configuration |
|:----------|:--------------|
| **Backbone** | Transformer (DiT-style) |
| **Layers** | 4 Transformer blocks |
| **Hidden Dimension** | 256 |
| **Attention Heads** | 4 |
| **Timesteps** | T = 1000 |
| **Noise Schedule** | Linear β: 1e-4 → 0.02 |
| **Conditioning** | AdaLN (Adaptive Layer Norm) |

```python
class TabularDiT(nn.Module):
    """Diffusion Transformer for tabular latent space."""
    
    def __init__(self, latent_dim=64, hidden_dim=256, n_layers=4, n_heads=4, n_classes=2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding (sinusoidal)
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)
        
        # Class embedding (for CFG)
        self.class_embed = nn.Embedding(n_classes + 1, hidden_dim)  # +1 for null class
        
        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
```

#### 4.1.3 Classifier-Free Guidance

| Parameter | Value | Description |
|:----------|:------|:------------|
| **Label Dropout Rate** | p = 0.1 | Probability of replacing label with null |
| **Guidance Scale (w)** | 2.0 (default) | Strength of class conditioning |
| **Null Token** | class_id = n_classes | Special embedding for unconditional |

```python
def cfg_sample(self, z_t, t, target_class, guidance_scale=2.0):
    """Classifier-Free Guidance sampling step."""
    
    # Conditional prediction
    eps_cond = self.model(z_t, t, class_label=target_class)
    
    # Unconditional prediction (null class)
    eps_uncond = self.model(z_t, t, class_label=None)
    
    # Guided prediction
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided
```

### 4.2 CTGAN (Baseline)

We use the reference implementation from the Synthetic Data Vault (SDV) library:

| Hyperparameter | Value |
|:---------------|:------|
| **Embedding Dimension** | 128 |
| **Generator Dimensions** | (256, 256) |
| **Discriminator Dimensions** | (256, 256) |
| **Generator Learning Rate** | 2e-4 |
| **Discriminator Learning Rate** | 2e-4 |
| **Batch Size** | 500 |
| **Epochs** | 100 |
| **Discriminator Steps** | 1 |
| **PAC** | 10 |

```python
from sdv.single_table import CTGANSynthesizer

ctgan = CTGANSynthesizer(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    batch_size=500,
    epochs=100,
    pac=10,
)
```

### 4.3 TVAE (Baseline)

| Hyperparameter | Value |
|:---------------|:------|
| **Embedding Dimension** | 128 |
| **Compress Dimensions** | (128, 128) |
| **Decompress Dimensions** | (128, 128) |
| **Learning Rate** | 1e-3 |
| **Batch Size** | 500 |
| **Epochs** | 100 |
| **Loss Factor** | 2.0 |

```python
from sdv.single_table import TVAESynthesizer

tvae = TVAESynthesizer(
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    l2scale=1e-5,
    batch_size=500,
    epochs=100,
    loss_factor=2.0,
)
```

### 4.4 Model Configuration Summary

| Model | Architecture | Parameters | Training Time (Adult) |
|:------|:-------------|:-----------|:----------------------|
| **RE-TabSyn** | VAE + DiT + CFG | ~1.2M | ~45 min |
| **CTGAN** | Conditional GAN | ~800K | ~20 min |
| **TVAE** | Variational AE | ~500K | ~15 min |

---

## 5. Training Procedures

### 5.1 RE-TabSyn Training Loop

Training proceeds in two phases:

#### Phase 1: VAE Training

```python
def train_vae(model, data, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in DataLoader(data, batch_size=256, shuffle=True):
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(batch)
            
            # Loss computation
            recon_loss = F.mse_loss(recon[:, :n_num], batch[:, :n_num])  # Numerical
            recon_loss += F.cross_entropy(recon[:, n_num:], batch[:, n_num:])  # Categorical
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + 0.1 * kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
```

#### Phase 2: Diffusion Training

```python
def train_diffusion(model, vae, data, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Freeze VAE
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs):
        for batch_x, batch_y in DataLoader(data, batch_size=256, shuffle=True):
            optimizer.zero_grad()
            
            # Encode to latent space
            with torch.no_grad():
                z_0 = vae.encode(batch_x)
            
            # Sample timestep
            t = torch.randint(0, T, (len(z_0),))
            
            # Add noise
            noise = torch.randn_like(z_0)
            z_t = sqrt_alphas_cumprod[t] * z_0 + sqrt_one_minus_alphas_cumprod[t] * noise
            
            # Label dropout for CFG (10% probability)
            mask = torch.rand(len(batch_y)) < 0.1
            batch_y_masked = batch_y.clone()
            batch_y_masked[mask] = NULL_CLASS
            
            # Predict noise
            noise_pred = model(z_t, t, batch_y_masked)
            
            # MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
```

### 5.2 Hyperparameter Summary

| Hyperparameter | RE-TabSyn | CTGAN | TVAE |
|:---------------|:----------|:------|:-----|
| **Batch Size** | 256 | 500 | 500 |
| **Learning Rate** | 1e-3 | 2e-4 | 1e-3 |
| **Optimizer** | Adam | Adam | Adam |
| **VAE Epochs** | 100 | - | 100 |
| **Diffusion Epochs** | 100 | - | - |
| **GAN Epochs** | - | 100 | - |
| **Weight Decay** | 0 | 0 | 1e-5 |
| **Gradient Clipping** | None | None | None |

### 5.3 Validation Strategy

We employ holdout validation rather than cross-validation due to computational constraints:

| Strategy | Details |
|:---------|:--------|
| **Train/Test Split** | 80% / 20% stratified |
| **Validation Metric** | VAE: Reconstruction loss; Diffusion: Noise prediction MSE |
| **Early Stopping** | Not used (fixed epochs for reproducibility) |
| **Model Selection** | Final epoch checkpoint |

**Justification:** Fixed training epochs ensure consistent comparison across methods and seeds. The generative setting lacks a clear validation objective analogous to supervised learning's validation accuracy.

---

## 6. Synthetic Data Generation Procedure

### 6.1 Generation Algorithm

```python
def generate(model, vae, num_samples, target_class=1, guidance_scale=2.0):
    """Generate synthetic samples with CFG."""
    
    # Step 1: Sample initial noise
    z_T = torch.randn(num_samples, latent_dim)
    
    # Step 2: Reverse diffusion with CFG
    z_t = z_T
    for t in reversed(range(T)):
        # CFG-guided noise prediction
        eps_cond = model(z_t, t, class_label=target_class)
        eps_uncond = model(z_t, t, class_label=None)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # Denoising step
        z_t = denoise_step(z_t, eps, t)
    
    z_0 = z_t
    
    # Step 3: Decode to data space
    synthetic = vae.decode(z_0)
    
    # Step 4: Post-processing
    synthetic = inverse_transform(synthetic)
    
    return synthetic
```

### 6.2 Post-Processing

| Step | Operation | Purpose |
|:-----|:----------|:--------|
| 1 | Inverse scaling | Restore numerical ranges |
| 2 | Categorical argmax | Convert logits to categories |
| 3 | Type casting | Ensure correct dtypes |
| 4 | Constraint enforcement | Clip to valid ranges |

### 6.3 Generation Parameters

| Parameter | Value | Effect |
|:----------|:------|:-------|
| **Num Samples** | len(test_set) | Match holdout size |
| **Target Class** | 1 (minority) | Generate minority-enriched |
| **Guidance Scale** | 2.0 | ~50% minority ratio |
| **Denoising Steps** | 1000 | Full reverse process |

---

## 7. Ablation Studies

We conduct ablation experiments to validate design choices:

### 7.1 Guidance Scale Ablation

| Guidance Scale (w) | Minority Ratio | KS Statistic | Interpretation |
|:-------------------|:---------------|:-------------|:---------------|
| 0.0 | 24.8% | 0.12 | Original distribution |
| 0.5 | 32.1% | 0.13 | Light guidance |
| 1.0 | 38.5% | 0.14 | Moderate guidance |
| **2.0** | **49.6%** | **0.15** | **Default (balanced)** |
| 3.0 | 58.2% | 0.18 | Over-guidance |
| 5.0 | 72.4% | 0.24 | Excessive minority |

**Observation:** Guidance scale w=2.0 achieves near-balanced classes (50%) with minimal fidelity degradation. Higher values produce majority-minority inversion but deteriorate distributional quality.

### 7.2 Backbone Ablation

| Backbone | KS Statistic | Training Time | Parameters |
|:---------|:-------------|:--------------|:-----------|
| MLP (3 layers) | 0.17 | 25 min | 400K |
| **Transformer (DiT)** | **0.15** | 45 min | 1.2M |
| U-Net (adapted) | 0.16 | 60 min | 2.1M |

**Observation:** Transformer backbone provides best fidelity, justifying the additional computational cost.

### 7.3 Latent Dimension Ablation

| Latent Dim | KS Statistic | Minority Ratio | Reconstruction Error |
|:-----------|:-------------|:---------------|:---------------------|
| 16 | 0.22 | 48.1% | 0.15 |
| 32 | 0.18 | 49.2% | 0.08 |
| **64** | **0.15** | **49.6%** | 0.05 |
| 128 | 0.15 | 49.8% | 0.04 |

**Observation:** Latent dimension 64 balances reconstruction quality and diffusion performance. Larger dimensions offer marginal improvement with increased computation.

### 7.4 VAE vs Direct Diffusion

| Approach | KS Statistic | Training Stability | Minority Preservation |
|:---------|:-------------|:-------------------|:----------------------|
| Direct Diffusion (TabDDPM-style) | 0.80 | Unstable | 0% (collapse) |
| **Latent Diffusion (RE-TabSyn)** | **0.15** | **Stable** | **50%** |

**Observation:** Latent space diffusion is essential for tabular data. Direct diffusion on one-hot encoded features fails catastrophically.

---

## 8. Computational Requirements

### 8.1 Training Time

| Dataset | RE-TabSyn | CTGAN | TVAE |
|:--------|:----------|:------|:-----|
| Adult (45K) | 45 min | 20 min | 15 min |
| German Credit (1K) | 10 min | 5 min | 3 min |
| Bank Marketing (41K) | 40 min | 18 min | 12 min |
| Credit Approval (690) | 8 min | 4 min | 3 min |
| Lending Club (10K) | 20 min | 10 min | 8 min |
| Polish Bankruptcy (5K) | 15 min | 8 min | 6 min |

### 8.2 Memory Usage

| Model | Peak GPU Memory | Peak CPU Memory |
|:------|:----------------|:----------------|
| RE-TabSyn | 2.1 GB | 4.3 GB |
| CTGAN | 1.8 GB | 3.2 GB |
| TVAE | 1.2 GB | 2.8 GB |

### 8.3 Total Experiment Time

| Configuration | Time |
|:--------------|:-----|
| Single run (1 dataset, 1 model, 1 seed) | ~45 min |
| Full benchmark (6 datasets × 3 models × 3 seeds) | ~27 hours |
| Ablation studies | ~15 hours |
| **Total** | **~42 hours** |

---

## 9. Code Availability

Implementation is available at: [Repository URL to be added upon publication]

```
codebase/
├── vae.py              # VAE implementation
├── latent_diffusion.py # Diffusion + CFG
├── transformer.py      # DiT backbone
├── models.py           # Unified wrapper
├── data_loader.py      # Dataset loading
├── evaluator.py        # Metrics computation
├── run_benchmark.py    # Single experiment
└── run_multi_benchmark.py  # Full benchmark
```

---

*Section word count: ~2,200*
