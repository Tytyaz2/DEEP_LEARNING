# Deep Learning Models Portfolio

A curated collection of deep learning generative model implementations, built from scratch in PyTorch using Jupyter Notebooks. Each project explores a different family of generative models, from classical autoencoders to modern diffusion networks.

## Projects Overview

| # | Project | Models | Dataset | Notebook |
|---|---------|--------|---------|----------|
| 1 | [Autoencoders](#1-autoencoders) | AE, VAE, VAE-CelebA | FashionMNIST, CelebA | `01_Autoencoders/` |
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

**Key concepts:** encoder/decoder, latent space, reparameterization trick, ELBO loss, KL divergence.

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
│   └── VariationalAutoEncoder_CelebA.ipynb
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
