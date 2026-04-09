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

### `VQVAE.ipynb` — Vector Quantized VAE
**Dataset:** FashionMNIST (28×28 grayscale, padded to 32×32)

**Architecture:**
- Encoder: 2 strided conv layers → 8×8 feature map of dimension D=64
- VectorQuantizer: codebook of K=512 vectors of dim D; nearest-neighbour lookup per spatial position
- Decoder: mirror with transposed convolutions → Tanh output

**Key innovation — straight-through estimator:** the `argmin` is non-differentiable, so gradients from the decoder are copied directly to the encoder as if quantization didn't exist:
```python
z_q = z_e + (z_q - z_e).detach()
```

**Loss (3 terms):** MSE reconstruction + codebook loss (moves `e` toward `z_e`) + β·commitment loss (moves `z_e` toward `e`)

**Training:** Adam (lr=2e-4), 20 epochs, batch size 128
**Compression:** 32×32 image → 8×8 = 64 discrete integers (each ∈ [0, 511]) — 16× compression

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
| Codebook | Discrete set of K learned vectors; each latent position maps to its nearest entry |
| Straight-through estimator | Trick to backpropagate through a non-differentiable argmin |
| Commitment loss | Prevents the encoder from oscillating between codebook entries |
| Codebook collapse | Failure mode where only a few codebook entries are ever used |
