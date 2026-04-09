# Flow-Based Models (Normalizing Flows)

> Exact likelihood generative models using sequences of invertible transformations.

## Notebook

### `flow_based_model.ipynb` — RealNVP

**Dataset:** 2D Moons (sklearn, 30,000 samples, noise=0.05, normalized)

## Model Architecture — RealNVP

RealNVP (Real-valued Non-Volume Preserving) stacks 4 **affine coupling layers** with alternating binary masks:

```
x = [x_A, x_B]   (split by mask)

Forward (x → z):
  z_A = x_A                              (identity)
  z_B = x_B · exp(s(x_A)) + t(x_A)      (scale + translate)

Inverse (z → x):
  x_A = z_A
  x_B = (z_B - t(z_A)) · exp(-s(z_A))
```

- `s` (scale) and `t` (translate) are **3-layer MLPs** with ReLU activations and hidden dim=256
- Masks alternate between [1,0] and [0,1] patterns across layers
- Alternating masks ensure all dimensions are transformed

**Prior:** Standard Gaussian N(0, I)

## Training

**Objective:** Maximum log-likelihood

```
log p(x) = log p_z(f(x)) + log |det J_f(x)|
         = log p_z(z) + Σ s_k(x_A)     (Jacobian is triangular → det = product of diagonals)
```

| Parameter | Value       |
|-----------|-------------|
| Coupling layers | 4           |
| Hidden dim | 256         |
| Batch size | 256         |
| Learning rate | 1e-4 (Adam) |
| Epochs | 50          |

**Loss:** −286 → −154 (negative log-likelihood, lower is better)

## Results

The model successfully learns the bimodal two-moon distribution. Generated samples reproduce the characteristic curved structure of the target distribution.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Normalizing flow | Chain of invertible functions mapping data ↔ latent |
| Change of variables | log p(x) = log p(z) + log \|det J_f\| |
| Coupling layer | Splits input; transforms one half conditioned on the other |
| Volume preservation | Jacobian determinant = product of scaling factors (triangular matrix) |
| Exact likelihood | Unlike VAEs, flows compute exact log p(x) — no ELBO approximation |
| Invertibility | Both sampling (z→x) and density evaluation (x→z) are tractable |
