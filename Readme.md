# `EOS LLM`

> **A 100M-parameter looped Transformer architecture exploring depth-recurrent weight sharing for extreme parameter efficiency.**

[![Status: In Active Development](https://img.shields.io/badge/Status-In_Development-0D9488?style=flat-square)](https://github.com/kivitas)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)

---

### Overview

Standard large language models scale performance by increasing total weight count across fixed depth. **`loop-100m`** explores an alternative paradigm: **depth-recurrent looping**. 

By reusing a core stack of Transformer layers across multiple execution loops ($k$-passes), the architecture achieves the effective expressive depth of a larger model while keeping the unique parameter footprint tightly constrained to **100M parameters**.

Inspired by recent advances in looped transformer architectures and recursive depth execution ([arXiv:2604.12946](https://arxiv.org/abs/2604.12946)), this project tests the limits of compute-vs-parameter trade-offs in small-footprint models.

---

### Architectural Highlights

* **100M Unique Weights / Variable Effective Depth:** Parameter-bound to ~100M unique weights, dynamically unrolled across $N$ recurrent iterations during forward passes.
* **Weight-Shared Block Recurrence:** Single or multi-layer attention/MLP blocks looped sequentially, enabling deeper latent reasoning per token without memory footprint inflation.
* **Stable Training Dynamics:** Implements specialized layer-norm placement, scale-decay mechanisms, and adaptive step-embeddings to prevent gradient vanishing/explosion across deep loop iterations.
* **Optimized for Edge AI:** Designed to fit entirely within low-memory micro-runtimes while leveraging variable compute at inference time.

---

### Core Loop Concept

$$\mathbf{h}_{t}^{(k)} = \text{TransformerBlock}\left(\mathbf{h}_{t}^{(k-1)}, \text{EMB}(k)\right)$$

Instead of passing hidden states through $L$ distinct layer weights:
1. Input embeddings enter the shared core block.
2. The hidden state $\mathbf{h}_t$ iterates through the **same** 100M parameter block for $K$ iterations.
3. Final representations are projected to vocabulary logits.

---

### Tech Stack & Tools

* **Framework:** PyTorch / PyTorch 2.x compile
* **Attention:** FlashAttention-2 / Rotary Position Embeddings (RoPE)
* **Optimization:** AdamW + Cosine Schedule with Warmup
* **Tokenizer:** Custom Byte-Pair Encoding (BPE)

---

### Current Status & Roadmap

- [x] Architecture design & dynamic loop forward pass implementation
- [x] Initial gradient flow & loop stability verification
- [ ] Pre-training dataset curation & tokenization
- [ ] 100M pre-training run on shared loop architecture
- [ ] Benchmark evaluations (Perplexity vs. standard 100M non-looped baseline)

---

### Author

Developed by **Kivitas** ([@kivitas](https://github.com/kivitas))  
*Building AI-native memory systems and parameter-efficient model architectures.*
