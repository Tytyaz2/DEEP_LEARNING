# Energy-Based Models

> Generative models that learn a scalar energy function E_θ(x), where low energy = high probability.

## Notebook

### `Energy_Based_Model.ipynb` — JEM (Joint Energy-Based Model)

**Dataset:** MNIST (10 classes, 28×28 grayscale)

## Model Architecture — JEMClassifier

A CNN with a dual-head output combining **generative** (energy) and **discriminative** (classification) objectives:

```
Input (28×28)
    │
Conv2D → Swish → Conv2D → Swish → ... (progressive downsampling to 2×2)
    │
    ├── FC head → energy score E_θ(x) ∈ ℝ     (generative)
    └── FC head → class logits f_θ(x) ∈ ℝ¹⁰   (discriminative)
```

**Activation:** Swish — x · sigmoid(x) — smoother than ReLU, better for energy landscape modeling.

**Connection between heads:**
```
p_θ(y | x) = softmax(f_θ(x))     [classifier]
p_θ(x) ∝ exp(−E_θ(x))            [generative model]
p_θ(x, y) = p_θ(y|x) · p_θ(x)   [joint model]
```

## Training — Multi-Component Loss

```
L_total = L_classification + L_contrastive + L_regularization

L_classification  = CrossEntropy(f_θ(x_real), y)
L_contrastive     = E_θ(x_fake) − E_θ(x_real)     (real should have lower energy)
L_regularization  = α · (E_θ(x_real)² + E_θ(x_fake)²)
```

## MCMC Sampling — Langevin Dynamics

The model generates negative samples x_fake via **Stochastic Gradient Langevin Dynamics (SGLD)**:

```
x_{t+1} = x_t − η/2 · ∇_x E_θ(x_t) + ε,   ε ~ N(0, η)
```

**Replay buffer:** 95% of samples come from a buffer of previously generated examples (warm starts), 5% from random noise. This stabilizes training and improves sample quality.

## Results

| Metric | Value |
|--------|-------|
| Epochs | 3 |
| Validation accuracy | 91.3% |

## Key Concepts

| Concept | Description |
|---------|-------------|
| Energy function | E_θ(x): scalar score — low energy = likely data |
| Gibbs distribution | p_θ(x) = exp(−E_θ(x)) / Z_θ, Z_θ intractable |
| Contrastive divergence | ∇L ≈ E[∇E(x_fake)] − E[∇E(x_real)] |
| MCMC / SGLD | Sampling from p_θ via gradient-guided random walk |
| Replay buffer | Stores past MCMC samples for warm-starting — reduces mixing time |
| JEM | Unified model: same network for classification AND generation |
