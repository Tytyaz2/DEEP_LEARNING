# Autoregressive Models

> Sequential generation by modeling p(x) = ∏ p(xᵢ | x₁, ..., xᵢ₋₁)

Two implementations of autoregressive generation: one for text (GPT-style transformer) and one for images (PixelCNN).

## Notebooks

### `auto_regressive_model.ipynb` — GPT for Text Generation
**Dataset:** Wine Reviews (Kaggle, 130k reviews)

**Architecture:**
- Token embeddings (dim=256) + positional embeddings
- 1 Transformer block: multi-head causal self-attention + FFN (Linear → ReLU → Linear)
- Layer normalization + residual connections
- Vocabulary: 15,000 most-frequent words + `<PAD>` and `<UNK>` tokens
- Causal masking ensures tokens only attend to past positions

**Training:** Adam (lr=1e-4), 100 epochs
**Loss:** 4.08 → 2.31 (cross-entropy on next-token prediction)

**Generation:** Autoregressive sampling with temperature τ=0.8 — the model continues wine critiques from a seed prompt.

---

### `Pixel_CNN.ipynb` — PixelCNN for Image Generation
**Dataset:** FashionMNIST (resized to 16×16, quantized to 4 grayscale levels)

**Architecture:**
- Initial 7×7 masked convolution (Type A — excludes current pixel)
- 5 residual blocks: Conv1×1 → MaskedConv3×3 (Type B) → Conv1×1
- Output: 4 logits per pixel (4-class classification)

**Masked Convolutions:**
- Type A: conditions only on pixels strictly before current position
- Type B: includes current pixel's previous channel contributions
- Enforces left-to-right, top-to-bottom causal ordering

**Training:** CrossEntropyLoss, Adam, 5 epochs. Loss: 0.44 → 0.32
**Generation:** Sequential pixel sampling — softmax → categorical sample → next pixel

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Autoregressive model | Factors joint distribution as a product of conditionals |
| Causal masking | Prevents attention to future positions (GPT) |
| Masked convolution | Convolution kernel zeroed for future pixels (PixelCNN) |
| Temperature sampling | Controls randomness: low T = conservative, high T = creative |
| Next-token prediction | Training objective: predict token at position t given 1..t-1 |
