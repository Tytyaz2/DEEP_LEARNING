# Deep Learning Models Portfolio

A curated collection of deep learning generative model implementations, built from scratch in PyTorch using Jupyter Notebooks. Each project explores a different family of generative models, from classical autoencoders to modern diffusion networks.

## Projects Overview

| # | Project | Models | Dataset | Notebook |
|---|---------|--------|---------|----------|
| 1 | [Autoencoders](#1-autoencoders) | AE, VAE, VAE-CelebA, VQVAE | FashionMNIST, CelebA | `01_Autoencoders/` |
| 2 | [Autoregressive Models](#2-autoregressive-models) | GPT, PixelCNN | Wine Reviews, FashionMNIST | `02_Autoregressive_Models/` |
| 3 | [Generative Adversarial Networks](#3-generative-adversarial-networks) | DCGAN, WGAN-GP, Conditional WGAN-GP | CelebA | `03_Generative_Adversarial_Networks/` |
| 4 | [Diffusion Models](#4-diffusion-models) | DDPM, DDIM (conditional) | Sprites 16x16 | `04_Diffusion_Models/` |
| 5 | [Flow-Based Models](#5-flow-based-models) | RealNVP | 2D Moons | `05_Flow_Based_Models/` |
| 6 | [Energy-Based Models](#6-energy-based-models) | JEM (Joint Energy Model) | MNIST | `06_Energy_Based_Models/` |

---

## 1. Autoencoders

> `01_Autoencoders/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/AutoEncodeur)

Three notebooks exploring the autoencoder family:

- **Autoencoder** — Convolutional AE on FashionMNIST with a 2D latent space; MSE reconstruction loss converges to ~0.033 after 10 epochs.
- **Variational Autoencoder (VAE)** — VAE with reparameterization trick (β=1.5) on FashionMNIST; combines BCE reconstruction + KL divergence.
- **VAE on CelebA** — Same VAE architecture scaled to 64×64 celebrity face images.
- **VQVAE** — Replaces the continuous latent space with a **discrete codebook** (K=512 vectors). Each 32×32 image is compressed to a 8×8 grid of integers via nearest-neighbour lookup. Training uses the straight-through estimator to backpropagate through the non-differentiable argmin. Foundation of DALL-E 1.

**Key concepts:** encoder/decoder, latent space, reparameterization trick, ELBO loss, KL divergence, vector quantization, codebook, straight-through estimator, commitment loss.

---

## 2. Autoregressive Models

> `02_Autoregressive_Models/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/auto_regressive_model)

Two notebooks implementing autoregressive generation:

- **GPT (text generation)** — Transformer decoder trained on 130k wine reviews (Kaggle). Vocabulary of 15,000 tokens, 256-dimensional embeddings, causal self-attention. Loss: 4.08 → 2.31 over 100 epochs. Generates wine critiques from a seed prompt.
- **PixelCNN (image generation)** — Masked convolutions on FashionMNIST (16×16, 4 gray levels). Residual blocks with Type A/B masked conv. Pixel-by-pixel sampling.

**Key concepts:** causal masking, self-attention, next-token prediction, masked convolutions, temperature sampling.

---

## 3. Generative Adversarial Networks

> `03_Generative_Adversarial_Networks/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/Deep_Convolutional_GAN)

Three GAN notebooks of increasing complexity:

- **DCGAN (grayscale)** — Deep convolutional GAN generating 64×64 grayscale images.
- **WGAN-GP** — Wasserstein GAN with gradient penalty on CelebA (64×64 RGB). Critic trained 5× per generator update, λ=10.
- **Conditional WGAN-GP** — Adds label conditioning (binary CelebA attributes) via one-hot concatenation to enable attribute-guided face generation.

**Key concepts:** adversarial training, Wasserstein distance, gradient penalty, Lipschitz constraint, conditional generation.

---

## 4. Diffusion Models

> `04_Diffusion_Models/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/Diffusion_model)

- **Conditional Diffusion Model (DDPM / DDIM)** — ContextUnet (U-Net with time & context embeddings) trained on 16×16 sprites. Linear noise schedule with 500 timesteps. DDIM accelerates sampling from 500 → ~25 steps. Context labels guide class-conditional generation.

**Key concepts:** forward/reverse diffusion process, noise scheduling, U-Net, DDPM, DDIM, classifier-free guidance.

---

## 5. Flow-Based Models

> `05_Flow_Based_Models/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/Flows_based_models)

- **RealNVP** — 4 stacked affine coupling layers with alternating binary masks, each parameterized by 3-layer MLPs. Trained on the 2D moons dataset (30k samples). Maximum likelihood training; loss 286 → 154 over 30 epochs. Successfully captures the bimodal distribution.

**Key concepts:** normalizing flows, bijective mappings, change of variables, log-likelihood, coupling layers, Jacobian determinant.

---

## 6. Energy-Based Models

> `06_Energy_Based_Models/` &nbsp;|&nbsp; [GitHub repo](https://github.com/Tytyaz2/Energy_based_model)

- **JEM (Joint Energy-Based Model)** — CNN with Swish activations on MNIST, dual-head architecture outputting both energy scores and class logits. Training combines contrastive divergence, regularization, and cross-entropy losses. MCMC sampling via Langevin dynamics with replay buffer (95% buffer / 5% noise). Achieves **91.3% validation accuracy** after 3 epochs.

**Key concepts:** energy functions, contrastive divergence, MCMC, Langevin dynamics, replay buffer, joint generative-discriminative training.

---

## When to Use Which Network?

Each family of generative models has its own strengths, weaknesses, and ideal use cases. This quick guide helps you pick the right tool for a given problem.

### Autoencoders (AE / VAE)

**Primary use:** learning a compact, structured representation of the data.

- **Use cases:** dimensionality reduction, image denoising, anomaly detection (abnormal samples have high reconstruction cost), interpolation between examples, recommendation systems.
- **VAE advantage:** the latent space is continuous and regularized → new samples can be drawn and interpolated coherently.
- **Limitations:** generated image quality is lower than GANs and diffusion models. A plain AE does not guarantee semantic structure in the latent space.

---

### Autoregressive Models (GPT, PixelCNN)

**Primary use:** modeling sequences by computing an exact probability over each element.

- **Use cases:** text generation (ChatGPT, Copilot), code completion, automatic captioning, pixel-by-pixel image generation (PixelCNN), symbolic music generation.
- **Key advantage:** exact probability p(x) — useful for lossless compression and model evaluation.
- **Limitations:** generation is **sequential** (one token at a time), so inference is slow and cannot be parallelized. Context is bounded by the attention window size.

---

### GANs (DCGAN, WGAN-GP, Conditional)

**Primary use:** generating highly realistic, high-resolution images.

- **Use cases:** face synthesis (StyleGAN), image super-resolution, data augmentation, artistic style transfer, attribute-guided generation (Conditional GAN).
- **Key advantage:** visual quality often surpasses other methods at high resolutions; inference is fast.
- **Limitations:** training is **unstable** and sensitive to hyperparameters, prone to *mode collapse* (generator converges to a single output type), no explicit log-likelihood making quantitative evaluation difficult.

---

### Diffusion Models (DDPM, DDIM)

**Primary use:** generating very high quality data with stable training.

- **Use cases:** image generation (Stable Diffusion, DALL-E 2, Midjourney), inpainting, denoising, audio synthesis (DiffWave), molecule generation in computational chemistry.
- **Key advantage:** current state-of-the-art image quality, **very stable** training compared to GANs, flexible conditioning (text, class label, image).
- **Limitations:** inference is **slow** — requires many denoising steps (mitigated by DDIM). High computational cost at both training and inference time.

---

### Flow-Based Models (RealNVP)

**Primary use:** computing exact probability density while supporting generation and perfect inversion.

- **Use cases:** density estimation, anomaly detection (via log-likelihood score), audio generation (WaveGlow, WaveFlow), data compression, physics and finance simulations.
- **Key advantage:** **exact likelihood** + **exact inversion** (x → z and z → x are both tractable) — something neither VAEs nor GANs can offer.
- **Limitations:** transformations must be **bijective** (strong architectural constraint), which limits expressiveness. Underperform GANs and diffusion models on complex high-resolution image data.

---

### Energy-Based Models (JEM)

**Primary use:** defining an implicit distribution through an energy function, unifying generation and classification in a single model.

- **Use cases:** adversarially robust classification, out-of-distribution (OOD) detection, data scoring and ranking, modeling physical systems (Boltzmann machines).
- **Key advantage:** **unified generative + discriminative** model — one network does both. Naturally robust to input perturbations.
- **Limitations:** MCMC sampling (Langevin dynamics) is **slow and hard to stabilize**. The partition function Z is intractable, complicating evaluation. Training is delicate and prone to divergence.

---

### Comparison Table

| Model | Exact likelihood | Generation speed | Image quality | Conditional control | Training stability |
|-------|:---:|:---:|:---:|:---:|:---:|
| Autoencoder / VAE | No (ELBO) | Fast | Moderate | Yes (VAE) | Stable |
| Autoregressive | **Yes** | Slow | Good | Yes | Stable |
| GAN | No | **Fast** | **Very high** | Yes | Unstable |
| Diffusion | No | Slow | **Very high** | **Very flexible** | **Very stable** |
| Flow-Based | **Yes** | Moderate | Moderate | Limited | Stable |
| Energy-Based | No | Very slow | Moderate | Yes | Difficult |

---

## Tech Stack

- **Framework:** PyTorch
- **Environment:** Jupyter Notebook
- **Datasets:** MNIST, FashionMNIST, CelebA, Wine Reviews (Kaggle), Sprites, 2D Moons (sklearn)

## Repository Structure

```
.
├── 01_Autoencoders/
│   ├── README.md
│   ├── Autoencoder.ipynb
│   ├── VariationalAutoEncoder.ipynb
│   ├── VariationalAutoEncoder_CelebA.ipynb
│   └── VQVAE.ipynb
├── 02_Autoregressive_Models/
│   ├── README.md
│   ├── auto_regressive_model.ipynb
│   └── Pixel_CNN.ipynb
├── 03_Generative_Adversarial_Networks/
│   ├── README.md
│   ├── Gan grey 64x64.ipynb
│   ├── WGAN-GP.ipynb
│   └── Conditional_WGAN-GP.ipynb
├── 04_Diffusion_Models/
│   ├── README.md
│   ├── Conditional_Diffusion_Model.ipynb
│   ├── Training.ipynb
│   ├── Sampling.ipynb
│   └── diffusion_utilities.py
├── 05_Flow_Based_Models/
│   ├── README.md
│   └── flow_based_model.ipynb
└── 06_Energy_Based_Models/
    ├── README.md
    └── Energy_Based_Model.ipynb
```
