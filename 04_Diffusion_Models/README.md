# Diffusion Models

> Score-based generative models that learn to reverse a Gaussian noise process.

Implementation of DDPM and DDIM with conditional generation via a ContextUnet architecture.

## Notebooks

### `Conditional_Diffusion_Model.ipynb` — Main implementation
### `Training.ipynb` — Training loop
### `Sampling.ipynb` — Inference / generation

**Dataset:** Sprites dataset (16×16 px, 1,788 samples) with 5-dimensional context labels.

## Model Architecture — ContextUnet

A U-Net conditioned on both timestep and context (class label):

```
Input (noisy image xₜ)
        │
   DownBlock 1 ──────────────────────────────┐
        │                                    │ skip connection
   DownBlock 2 ────────────────────────┐     │
        │                              │     │
   Bottleneck                          │     │
        │  ← time embedding            │     │
        │  ← context embedding         │     │
   UpBlock 1  ←───────────────────────┘     │
        │                                    │
   UpBlock 2  ←──────────────────────────────┘
        │
   Output (predicted noise ε̂)
```

Each block: Conv2D → GroupNorm → GELU

## Diffusion Process

**Forward process** (training): gradually add Gaussian noise over T=500 timesteps with linear schedule β₁=1e⁻⁴ → β_T=0.02

**Reverse process** (inference):
- **DDPM**: iteratively denoise over all 500 steps
- **DDIM**: deterministic sampling — reduces steps from 500 → ~25 (20× speedup)

**Loss:** MSE between predicted noise ε̂ and actual noise ε added at timestep t

## Key Concepts

| Concept | Description |
|---------|-------------|
| Forward process | q(xₜ\|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI) — adds noise |
| Reverse process | p_θ(xₜ₋₁\|xₜ) — learned denoising step |
| Noise schedule | Controls how fast signal is destroyed: linear βₜ here |
| DDPM | Stochastic reverse: full T steps |
| DDIM | Deterministic reverse: skip steps, same quality |
| Conditioning | Time & context embeddings injected at bottleneck |
