# Methodology

## 1. Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ denote a tabular dataset where $\mathbf{x}_i \in \mathcal{X}$ represents a feature vector and $y_i \in \{0, 1\}$ denotes the class label. Each feature vector comprises numerical and categorical attributes:

$$\mathbf{x} = (\mathbf{x}^{(num)}, \mathbf{x}^{(cat)}) \in \mathbb{R}^{d_{num}} \times \prod_{j=1}^{d_{cat}} \mathcal{C}_j$$

where $\mathcal{C}_j$ is the categorical domain for attribute $j$ with cardinality $|\mathcal{C}_j|$.

**Objective:** Learn a generative model $G_\theta$ that synthesizes samples $\tilde{\mathbf{x}} \sim p_\theta(\mathbf{x})$ indistinguishable from $\mathbf{x} \sim p_{data}(\mathbf{x})$, with the additional capability to control the class distribution $p_\theta(y)$.

**Desiderata:**
1. **Fidelity:** $p_\theta(\mathbf{x}) \approx p_{data}(\mathbf{x})$ (distributional similarity)
2. **Utility:** Classifiers trained on synthetic data perform comparably on real data
3. **Privacy:** Synthetic samples do not memorize training instances
4. **Controllability:** User-specified class ratio $\pi = P(y=1)$ achievable

---

## 2. Baseline Methods: Theoretical Foundations

### 2.1 Generative Adversarial Networks (GANs)

GANs formulate generative modeling as a two-player minimax game between a generator $G$ and discriminator $D$:

$$\min_G \max_D \mathcal{L}_{GAN}(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

where $\mathbf{z} \sim p_z(\mathbf{z})$ is a latent noise vector, typically $\mathbf{z} \sim \mathcal{N}(0, I)$.

**Optimal Discriminator:** For fixed $G$, the optimal discriminator is:

$$D^*_G(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}$$

**Generator Objective:** Under optimal $D$, the generator minimizes the Jensen-Shannon divergence:

$$\mathcal{L}_G = 2 \cdot JSD(p_{data} \| p_g) - \log 4$$

#### 2.1.1 CTGAN: Conditional Tabular GAN

CTGAN (Xu et al., 2019) extends vanilla GANs for tabular data through three mechanisms:

**Mode-Specific Normalization.** For multimodal numerical features, CTGAN fits a Gaussian Mixture Model (GMM) and normalizes within each mode:

$$\mathbf{x}^{(num)}_j \rightarrow (\alpha_{j,k}, \beta_{j,k})$$

where $\alpha_{j,k} = \frac{x_j - \mu_k}{\sigma_k}$ is the normalized value and $\beta_{j,k}$ is a one-hot indicator of the selected mode $k$.

**Conditional Generator.** The generator conditions on a discrete column value $c$:

$$G(\mathbf{z}, c) \rightarrow \tilde{\mathbf{x}}$$

This ensures balanced sampling across categorical values during training.

**Training-by-Sampling.** Each training batch samples conditions uniformly:

$$c \sim \text{Uniform}(\mathcal{C}), \quad \mathbf{x} \sim p_{data}(\mathbf{x} | c)$$

**CTGAN Loss Functions:**

Generator:
$$\mathcal{L}_G = -\mathbb{E}_{\mathbf{z}, c}[\log D(G(\mathbf{z}, c), c)] + \lambda \cdot \mathcal{L}_{cond}$$

where $\mathcal{L}_{cond}$ penalizes mismatch between generated and conditioned columns.

Discriminator:
$$\mathcal{L}_D = -\mathbb{E}_{\mathbf{x}, c}[\log D(\mathbf{x}, c)] - \mathbb{E}_{\mathbf{z}, c}[\log(1 - D(G(\mathbf{z}, c), c))]$$

**Limitations:** Mode collapse on minority classes; training instability; no mechanism for explicit class ratio control.

---

### 2.2 Variational Autoencoders (VAEs)

VAEs model data through latent variables $\mathbf{z}$ with:
- Prior: $p(\mathbf{z}) = \mathcal{N}(0, I)$
- Likelihood: $p_\theta(\mathbf{x}|\mathbf{z})$ (decoder)
- Posterior: $q_\phi(\mathbf{z}|\mathbf{x})$ (encoder)

**Evidence Lower Bound (ELBO):**

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$$\mathcal{L}_{VAE} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}$$

**Reparameterization Trick:** For $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x}))$:

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This enables gradient-based optimization through the stochastic sampling operation.

#### 2.2.1 TVAE: Tabular VAE

TVAE applies VAE principles to tabular data with mixed-type reconstruction:

$$\mathcal{L}_{recon} = \sum_{j \in num} \|\mathbf{x}_j - \hat{\mathbf{x}}_j\|^2 + \sum_{j \in cat} \text{CrossEntropy}(\mathbf{x}_j, \hat{\mathbf{x}}_j)$$

**Generation:**
$$\mathbf{z} \sim \mathcal{N}(0, I), \quad \tilde{\mathbf{x}} = \text{Decoder}_\theta(\mathbf{z})$$

**Limitations:** Blurry reconstructions; posterior collapse; no class conditioning mechanism.

---

### 2.3 Diffusion Models

Diffusion models define a forward noising process and learn to reverse it.

#### 2.3.1 Forward Process

The forward process gradually adds Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t I)$$

where $\{\beta_t\}_{t=1}^T$ is a variance schedule.

Using the notation $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, we can sample $\mathbf{x}_t$ directly:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)I)$$

Equivalently:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 2.3.2 Reverse Process

The reverse process learns to denoise:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

**Noise Prediction Formulation:** Instead of predicting $\mu_\theta$, we predict the noise $\epsilon_\theta$:

$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right)$$

**Training Objective:**

$$\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2\right]$$

#### 2.3.3 TabDDPM: Diffusion for Tabular Data

TabDDPM applies diffusion to tabular data by:

1. **Gaussian Diffusion** for numerical features
2. **Multinomial Diffusion** for categorical features:

$$q(\mathbf{x}_t^{(cat)} | \mathbf{x}_0^{(cat)}) = \text{Cat}(\mathbf{x}_t; \mathbf{p} = \bar{\alpha}_t \mathbf{x}_0 + (1-\bar{\alpha}_t)/K)$$

where $K$ is the number of categories.

**Limitation:** Direct diffusion on one-hot encoded features creates sparse, discontinuous gradients leading to poor convergence (KS > 0.80 in our experiments).

---

## 3. Proposed Method: RE-TabSyn

RE-TabSyn (Rare-Event Enhanced Tabular Synthesis) combines three components:
1. **Variational Autoencoder** for mixed-type encoding
2. **Latent Diffusion Model** for high-quality generation
3. **Classifier-Free Guidance** for controllable class ratios

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RE-TabSyn Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Training Phase:                                                         │
│  ┌──────────┐      ┌──────────┐      ┌──────────────────┐               │
│  │  Mixed   │──────│   VAE    │──────│ Latent Diffusion │               │
│  │  Table   │      │ Encoding │      │  with CFG        │               │
│  │   x      │      │    z     │      │  Training        │               │
│  └──────────┘      └──────────┘      └──────────────────┘               │
│                                                                          │
│  Generation Phase:                                                       │
│  ┌──────────┐      ┌──────────────────┐      ┌──────────┐  ┌──────────┐│
│  │  Noise   │──────│ CFG-Guided       │──────│   VAE    │──│ Synthetic││
│  │  z_T     │      │ Reverse Diffusion│      │ Decoding │  │  Table   ││
│  └──────────┘      └──────────────────┘      └──────────┘  └──────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component 1: Tabular VAE

#### 3.2.1 Encoder

The encoder maps mixed-type input $\mathbf{x} \in \mathbb{R}^{d}$ to latent parameters:

$$\mathbf{h} = \text{ReLU}(\text{LayerNorm}(W_1 \mathbf{x} + b_1))$$
$$\mu = W_\mu \mathbf{h} + b_\mu, \quad \log\sigma^2 = W_\sigma \mathbf{h} + b_\sigma$$

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \text{diag}(\sigma^2_\phi(\mathbf{x})))$$

#### 3.2.2 Decoder

The decoder reconstructs from latent:

$$\hat{\mathbf{x}}^{(num)} = W_{num}\text{ReLU}(W_h\mathbf{z} + b_h) + b_{num}$$
$$\hat{\mathbf{x}}^{(cat)}_j = \text{Softmax}(W_{cat,j}\text{ReLU}(W_h\mathbf{z} + b_h) + b_{cat,j})$$

#### 3.2.3 Loss Function

$$\mathcal{L}_{VAE} = \underbrace{\sum_{j=1}^{d_{num}} \|\mathbf{x}_j - \hat{\mathbf{x}}_j\|^2}_{\text{Numerical MSE}} + \underbrace{\sum_{j=1}^{d_{cat}} \text{CE}(\mathbf{x}_j, \hat{\mathbf{x}}_j)}_{\text{Categorical Cross-Entropy}} + \underbrace{\beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL Regularization}}$$

where $\beta = 0.1$ balances reconstruction and regularization.

**KL Divergence (closed form):**
$$D_{KL} = -\frac{1}{2}\sum_{i=1}^{d_z}\left(1 + \log\sigma^2_i - \mu^2_i - \sigma^2_i\right)$$

### 3.3 Component 2: Latent Diffusion Model

#### 3.3.1 Forward Process in Latent Space

Given encoded latent $\mathbf{z}_0 = \text{Encoder}(\mathbf{x})$:

$$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Noise Schedule:** Linear schedule with $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$.

#### 3.3.2 Conditional Denoising Network

We employ a Transformer-based architecture for noise prediction:

$$\epsilon_\theta(\mathbf{z}_t, t, y) = \text{DiT}(\mathbf{z}_t, \text{TimeEmb}(t), \text{ClassEmb}(y))$$

**Diffusion Transformer (DiT) Block:**

$$\mathbf{h} = \mathbf{z}_t + \text{Attention}(\text{AdaLN}(\mathbf{z}_t, t, y))$$
$$\mathbf{o} = \mathbf{h} + \text{MLP}(\text{AdaLN}(\mathbf{h}, t, y))$$

**Adaptive Layer Normalization (AdaLN):**

$$(\gamma, \beta) = \text{MLP}([\text{TimeEmb}(t); \text{ClassEmb}(y)])$$
$$\text{AdaLN}(\mathbf{h}) = \gamma \odot \text{LayerNorm}(\mathbf{h}) + \beta$$

#### 3.3.3 Training Objective

$$\mathcal{L}_{diff} = \mathbb{E}_{t \sim U[1,T], \mathbf{z}_0, \epsilon, y}\left[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, y)\|^2\right]$$

### 3.4 Component 3: Classifier-Free Guidance (CFG)

CFG enables conditional generation without a separate classifier by jointly training conditional and unconditional models.

#### 3.4.1 Training with Label Dropout

During training, class labels are randomly replaced with a null token $\varnothing$ with probability $p_{uncond} = 0.1$:

$$\tilde{y} = \begin{cases} y & \text{with probability } 1 - p_{uncond} \\ \varnothing & \text{with probability } p_{uncond} \end{cases}$$

This trains the network for both:
- **Conditional:** $\epsilon_\theta(\mathbf{z}_t, t, y)$
- **Unconditional:** $\epsilon_\theta(\mathbf{z}_t, t, \varnothing)$

#### 3.4.2 Guided Sampling

At generation time, the noise prediction is interpolated:

$$\tilde{\epsilon}_\theta(\mathbf{z}_t, t, y) = \underbrace{\epsilon_\theta(\mathbf{z}_t, t, \varnothing)}_{\text{unconditional}} + w \cdot \underbrace{\left(\epsilon_\theta(\mathbf{z}_t, t, y) - \epsilon_\theta(\mathbf{z}_t, t, \varnothing)\right)}_{\text{conditional direction}}$$

where $w \geq 0$ is the **guidance scale**:
- $w = 0$: Unconditional generation (original class distribution)
- $w = 1$: Standard conditional generation
- $w > 1$: Stronger class adherence (minority boosting)

**Default:** $w = 2.0$ achieves ~50% minority ratio from ~25% original.

#### 3.4.3 Mathematical Interpretation

CFG can be interpreted as modifying the score function:

$$\nabla_{\mathbf{z}} \log p(\mathbf{z}|y) \propto \nabla_{\mathbf{z}} \log p(\mathbf{z}) + w \cdot \nabla_{\mathbf{z}} \log p(y|\mathbf{z})$$

The guided score emphasizes regions where class $y$ is likely, effectively upweighting minority class density.

### 3.5 Complete Algorithm

**Algorithm 1: RE-TabSyn Training**

```
Input: Dataset D = {(x_i, y_i)}, VAE epochs E_vae, Diffusion epochs E_diff
Output: Trained VAE (φ, θ), Diffusion model ψ

// Phase 1: Train VAE
for epoch = 1 to E_vae do
    for batch (x, y) in D do
        μ, log σ² = Encoder_φ(x)
        z = μ + σ ⊙ ε,  ε ~ N(0, I)
        x̂ = Decoder_θ(z)
        L = L_recon(x, x̂) + β · KL(q_φ(z|x) || p(z))
        Update (φ, θ) via gradient descent on L
    end for
end for

// Phase 2: Train Diffusion with CFG
Freeze VAE parameters
for epoch = 1 to E_diff do
    for batch (x, y) in D do
        z_0 = Encoder_φ(x)
        t ~ Uniform{1, ..., T}
        ε ~ N(0, I)
        z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
        
        // Label dropout for CFG
        ỹ = y with prob 0.9, else ∅
        
        ε̂ = DiT_ψ(z_t, t, ỹ)
        L = ||ε - ε̂||²
        Update ψ via gradient descent on L
    end for
end for

return (φ, θ, ψ)
```

**Algorithm 2: RE-TabSyn Generation with CFG**

```
Input: Trained models (φ, θ, ψ), num_samples N, target_class y*, guidance_scale w
Output: Synthetic samples X̃

z_T ~ N(0, I)^N  // Initialize with noise

// Reverse diffusion with CFG
for t = T to 1 do
    ε_uncond = DiT_ψ(z_t, t, ∅)
    ε_cond = DiT_ψ(z_t, t, y*)
    ε̃ = ε_uncond + w · (ε_cond - ε_uncond)
    
    z_{t-1} = (1/√α_t)(z_t - (β_t/√(1-ᾱ_t))ε̃) + σ_t · ε,  ε ~ N(0, I)
end for

// Decode to data space
X̃ = Decoder_θ(z_0)

// Post-process
X̃_num = InverseScale(X̃_num)
X̃_cat = Argmax(X̃_cat)

return X̃
```

---

## 4. Data Encoding for Mixed-Type Features

### 4.1 Preprocessing Pipeline

**Numerical Features:**
$$x_j^{(num)} \xrightarrow{\text{QuantileTransform}} \tilde{x}_j \sim \mathcal{N}(0, 1)$$

The quantile transformer maps arbitrary distributions to standard Gaussian, enabling stable VAE training.

**Categorical Features:**
$$x_j^{(cat)} \in \{c_1, ..., c_K\} \xrightarrow{\text{LabelEncode}} \tilde{x}_j \in \{0, 1, ..., K-1\}$$

Label encoding preserves ordinality where applicable and reduces dimensionality versus one-hot encoding.

### 4.2 Decoder Output Heads

**Numerical Reconstruction:**
$$\hat{x}_j^{(num)} = W_j^{(num)} \mathbf{h} + b_j^{(num)} \in \mathbb{R}$$

Loss: Mean Squared Error

**Categorical Reconstruction:**
$$\hat{x}_j^{(cat)} = \text{Softmax}(W_j^{(cat)} \mathbf{h} + b_j^{(cat)}) \in \Delta^{K-1}$$

Loss: Cross-Entropy

### 4.3 Inverse Transform

$$\tilde{x}^{(num)} \xrightarrow{\text{InverseQuantile}} x^{(num)}$$
$$\tilde{x}^{(cat)} \xrightarrow{\text{InverseLabelEncode}} x^{(cat)}$$

---

## 5. Theoretical Analysis

### 5.1 Why Latent Diffusion for Tabular Data?

**Proposition 1.** *Direct diffusion on one-hot encoded categorical features induces discontinuous gradients that impede optimization.*

*Sketch:* One-hot vectors lie on the vertices of a hypercube $\{0,1\}^K$. Adding Gaussian noise moves points into the continuous interior, but the reconstruction target remains discrete. The gradient landscape exhibits sharp discontinuities at decision boundaries.

**Proposition 2.** *VAE encoding maps mixed-type tabular data to a continuous latent space amenable to diffusion.*

*Sketch:* The VAE encoder learns a smooth mapping $\mathbf{x} \mapsto \mathbf{z}$ where nearby latent points decode to semantically similar table rows. This continuity enables stable diffusion training and generation.

### 5.2 CFG for Class Distribution Control

**Proposition 3.** *Classifier-Free Guidance with scale $w > 1$ increases the sampling probability of minority class instances.*

*Proof Sketch:* The guided score function becomes:

$$\tilde{s}_\theta(\mathbf{z}_t, t, y) = (1-w)s_\theta(\mathbf{z}_t, t) + w \cdot s_\theta(\mathbf{z}_t, t, y)$$

For minority class $y = 1$, this upweights the conditional score, effectively sampling from a reweighted distribution:

$$\tilde{p}(\mathbf{z} | y=1) \propto p(\mathbf{z})^{1-w} \cdot p(\mathbf{z}|y=1)^w$$

As $w$ increases, the sampled distribution concentrates more on minority-conditioned regions.

---

## 6. Model Architecture Details

### 6.1 VAE Architecture

| Layer | Input Dim | Output Dim | Activation |
|:------|:----------|:-----------|:-----------|
| Encoder Linear 1 | $d_{input}$ | 256 | ReLU |
| Encoder Linear 2 | 256 | 128 | ReLU |
| Mean Head | 128 | $d_{latent}$ | None |
| LogVar Head | 128 | $d_{latent}$ | None |
| Decoder Linear 1 | $d_{latent}$ | 128 | ReLU |
| Decoder Linear 2 | 128 | 256 | ReLU |
| Output Linear | 256 | $d_{output}$ | Mixed |

### 6.2 Diffusion Transformer (DiT)

| Component | Configuration |
|:----------|:--------------|
| Input Projection | Linear($d_{latent}$, 256) |
| Time Embedding | Sinusoidal(256) → MLP → 256 |
| Class Embedding | Embedding($n_{classes}+1$, 256) |
| Transformer Blocks | 4 × DiTBlock(256, heads=4) |
| Output Projection | Linear(256, $d_{latent}$) |

**DiT Block:**
```
Input z_t, time t, class y
    → AdaLN(z_t, t, y)
    → MultiHeadAttention(4 heads)
    → Residual Connection
    → AdaLN
    → MLP(256 → 1024 → 256)
    → Residual Connection
→ Output
```

---

## 7. Assumptions and Constraints

### 7.1 Assumptions

1. **Stationarity:** The data distribution is stationary (no concept drift).
2. **Independence:** Rows are independently sampled from the population.
3. **Missing at Random (MAR):** Missing values are MAR for imputation validity.
4. **Class Label Availability:** Class labels are known for all training samples.

### 7.2 Constraints

1. **Binary Classification:** Current implementation supports binary targets; extension to multi-class is straightforward.
2. **Single Table:** Relational/multi-table data not supported.
3. **Moderate Dimensionality:** Tested on $d \leq 64$ features; scalability to very high dimensions unverified.
4. **Sufficient Minority Samples:** Requires minimum minority class samples for CFG training (empirically $n_{min} \geq 100$).

### 7.3 Computational Complexity

| Phase | Time Complexity | Space Complexity |
|:------|:----------------|:-----------------|
| VAE Training | $O(E_{vae} \cdot N \cdot d \cdot d_z)$ | $O(d \cdot d_z)$ |
| Diffusion Training | $O(E_{diff} \cdot N \cdot d_z^2)$ | $O(d_z^2)$ |
| Generation | $O(n \cdot T \cdot d_z^2)$ | $O(n \cdot d_z)$ |

where $N$ = samples, $d$ = features, $d_z$ = latent dim, $T$ = diffusion steps, $n$ = generated samples.

---

*Section word count: ~2,500*
