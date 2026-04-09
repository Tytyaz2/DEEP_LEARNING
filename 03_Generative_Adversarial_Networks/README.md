# Generative Adversarial Networks

> Implicit generative models trained via adversarial min-max game between a generator and a discriminator/critic.

Three notebooks of increasing complexity, from basic DCGAN to conditional Wasserstein GAN.

## Notebooks

### `Gan grey 64x64.ipynb` — Deep Convolutional GAN (Grayscale)
Basic DCGAN generating 64×64 grayscale images. Demonstrates the standard GAN training loop with transposed convolution upsampling in the generator and strided convolution downsampling in the discriminator.

---

### `WGAN-GP.ipynb` — Wasserstein GAN with Gradient Penalty
**Dataset:** CelebA (64×64 RGB face images, normalized to [-1, 1])

**Architecture:**
- Generator: z ∈ ℝ¹²⁸ → transposed convolutions with BatchNorm → 64×64 RGB (Tanh)
- Critic: strided convolutions with LeakyReLU → scalar score (no sigmoid)

**Training:**
- 5 critic updates per generator update
- Gradient penalty weight λ = 10 to enforce Lipschitz constraint
- Adam (lr=0.0002), 10 epochs, batch size 128

**Why WGAN?** Uses Wasserstein-1 distance instead of JS divergence → smoother gradients, more stable training, no mode collapse.

---

### `Conditional_WGAN-GP.ipynb` — Conditional WGAN-GP
**Dataset:** CelebA with binary attribute labels (e.g., `Blond_Hair`)

**Conditioning:** One-hot encoded attribute labels are spatially expanded and concatenated to both:
- Generator input (alongside noise vector z)
- Critic input (alongside real/fake images)

This enables **attribute-guided face generation**: the model can be steered to generate images with specific features (blond hair, smile, glasses, etc.).

The core WGAN-GP training loop (gradient penalty, critic iterations) remains unchanged — conditioning is purely architectural.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Generator | Maps latent noise z → synthetic data |
| Discriminator / Critic | Distinguishes real from fake (WGAN: outputs a score, not a probability) |
| Adversarial loss | min_G max_D E[D(x)] − E[D(G(z))] |
| Wasserstein distance | Earth-mover distance — more stable training signal than JS divergence |
| Gradient penalty | Penalizes ‖∇D‖ ≠ 1 on interpolated samples to enforce Lipschitz constraint |
| Conditional GAN | Conditions both G and D on auxiliary label y for class-guided generation |
| Mode collapse | GAN failure mode where G maps all z to same output — mitigated by WGAN |
