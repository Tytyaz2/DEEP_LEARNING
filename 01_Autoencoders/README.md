# Autoencoders

> Unsupervised representation learning through encoder-decoder architectures.

Three notebooks exploring the autoencoder family, from a basic reconstruction model to variational generation on celebrity faces.

## Notebooks

### `Autoencoder.ipynb` — Convolutional Autoencoder
**Dataset:** FashionMNIST (28×28 grayscale, padded to 32×32)

**Architecture:**
- Encoder: 3 convolutional layers (1→32→64→128 channels), stride-2 downsampling
- Latent space: fully-connected bottleneck (default 2D for visualization)
- Decoder: mirror architecture with transposed convolutions + Sigmoid

**Training:** MSE loss, Adam (lr=0.001), 10 epochs, batch size 64
**Results:** MSE converges to ~0.033. The 2D latent space allows direct visualization of class clusters.

---

### `VariationalAutoEncoder.ipynb` — VAE on FashionMNIST
**Dataset:** FashionMNIST (padded to 32×32)

**Architecture:**
- Same convolutional backbone as AE
- Two FC heads: one for μ (mean), one for log σ² (log-variance)
- Reparameterization: z = μ + σ · ε, ε ~ N(0, I)
- Decoder: transposed convolutions → Sigmoid

**Loss:** ELBO = BCE reconstruction loss + β·KL divergence (β = 1.5)
**Training:** Adam (lr=1e-4), 3 epochs, batch size 64

---

### `VariationalAutoEncoder_CelebA.ipynb` — VAE on CelebA
**Dataset:** CelebA (64×64 RGB celebrity faces)

Same VAE architecture adapted for higher-resolution color images. Demonstrates scalability of the VAE framework to real-world face data.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Encoder | Compresses input x into a latent representation z |
| Decoder | Reconstructs x̂ from z |
| Reparameterization trick | Enables backprop through stochastic sampling |
| ELBO | Evidence Lower BOund = reconstruction − KL(q(z\|x) \|\| p(z)) |
| β-VAE | Controls disentanglement vs. reconstruction quality trade-off |
| Latent space | Low-dimensional manifold capturing data structure |
