# 🌌 EOS Looped LLM: State-of-the-Art Recurrent-Depth Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Architecture: Recurrent MoE](https://img.shields.io/badge/Architecture-Recurrent%20MoE%20--%20MLA-blueviolet.svg)]()
[![Context: 16k Tokens](https://img.shields.io/badge/Context-16384%20Tokens-success.svg)]()

> **EOS Looped LLM** is a highly efficient, closed-source Large Language Model architecture designed for maximum hardware performance, extreme memory compression, and deep reasoning. By pairing a **parameter-shared recurrent core** with **Multi-Latent Attention (MLA)**, **Sparse Mixture-of-Experts (MoE)**, and **Hybrid Muon Optimization**, EOS computes at an effective depth of **50 Transformer Layers** using only **114M parameters**.

---

## 📌 Architecture Overview

```mermaid
graph TD
    A["Input Token Stream (16k Context)"] --> B["Token Embeddings & YaRN RoPE"]
    B --> C["Pre-Loop Transformer Layer (Layer 0)"]
    C --> D["Parameter-Shared Recurrent Core (Layers 1-4)"]
    
    subgraph Recurrent Loop Core [Recurrent Core: Up to 12 Loops]
        D1["FiLM Step Modulation"] --> D2["Multi-Latent Attention (MLA + QK-Norm)"]
        D2 --> D3["GroupedGEMM Sparse MoE (Top-2 / 4 Experts)"]
        D3 --> D4["Damped Anderson Fixed-Point Gating"]
        D4 --> D5{"Entropy-Gated ACT / Loop Bypass"}
    end
    
    D --> Recurrent Loop Core
    Recurrent Loop Core -- "Halt / Exit" --> E["Post-Loop Transformer Layer (Layer 5)"]
    E --> F["RMSNorm & Gemma-2 Logit Soft-Capping"]
    F --> G["True Chunked Cross-Entropy / LM Head"]
```

---

## 🚀 Key Engineering Innovations

### 1. Recurrent Depth Scaling (50 Effective Layers)
Instead of allocating 500M+ physical parameters, EOS shares a 4-layer Transformer core across up to **12 recurrent loops**:
$$\text{Effective Depth} = 1\text{ pre-layer} + (4\text{ shared layers} \times 12\text{ loops}) + 1\text{ post-layer} = \mathbf{50\text{ Layers}}$$
This delivers the expressive reasoning depth of a **~500M parameter model** while occupying only **114M parameters** in VRAM.

### 2. Multi-Latent Attention (MLA) & 4x KV Compression
Key-Value caches are compressed into low-rank latent vectors $\mathbf{c}_t^{KV}$ ($140$ dimensions vs $560$ full projection size).
* **4x VRAM Reduction:** Enables processing **16,384 token context windows** with minimal memory footprint.
* **QK-Norm:** Applies RMSNorm to Queries and Keys prior to dot-product attention to eliminate attention entropy collapse.
* **YaRN RoPE Extension:** Integrates frequency-dependent interpolation ($\theta = 500,000.0$) for long-context stability.

### 3. Gemma-2 Style Logit Soft-Capping
Applies bounded $\tanh$ soft-capping to attention scores ($\text{cap}=50.0$) and final LM head logits ($\text{cap}=30.0$):
$$\text{Scores}_{\text{capped}} = \text{cap} \cdot \tanh\left(\frac{\text{Scores}}{\text{cap}}\right)$$
Prevents gradient spikes, logit explosion, and numerical instability in FP16/BF16 precision.

### 4. Zero-Padding Document Multipacking
Replaces padding zeros (`<pad>`) with continuous document stream wrap-around. 
* **100% Compute Utilization:** Zero wasted padding tokens during pre-training.
* **Block-Diagonal Attention Masking:** Enforces strict boundary insulation between packed documents.

### 5. Hybrid Muon + AdamW Optimizer
* **Muon Group (2D Linear Matrices):** Applies orthogonalized updates via 5th-order Newton-Schulz iterations with symmetric aspect-ratio scaling:
  $$\alpha = \text{lr} \times \sqrt{\frac{\max(\text{rows}, \text{cols})}{\min(\text{rows}, \text{cols})}}$$
* **AdamW Group (Embeddings, Norms, Biases):** Handles 1D and embedding parameters with decoupled weight decay.

---

## 📐 Mathematical Specifications & Losses

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \min\left(0.15 \cdot \mathcal{L}_{\text{CE}},\, \mathcal{L}_{\text{ponder}} + \mathcal{L}_{\text{moe}} + \mathcal{L}_{\text{prior}} + \lambda_{\text{contract}} \mathcal{L}_{\text{contract}}\right) + \lambda_z \mathcal{L}_z$$

Where:
* $\mathcal{L}_{\text{CE}}$: True Chunked Cross-Entropy Loss (projected lazily per-chunk without materializing $[B, S, V]$).
* $\mathcal{L}_{\text{contract}}$: Banach Fixed-Point Contraction Loss $\frac{1}{K-1} \sum_{k=1}^{K-1} \|h_{k+1} - h_k\|_2^2$.
* $\mathcal{L}_{\text{moe}}$: Vectorized Load-Balancing Loss across MoE experts.
* $\mathcal{L}_z$: Logit $z$-loss $\text{mean}(\log^2 \sum e^{z_i})$ to prevent logit drift.

---

## 📂 Repository Structure

```
EOS LLM/
 ├── config.py                 # LoopedLLMConfig specification
 ├── model.py                  # EOSLoopedLLM architecture & attention
 ├── train.py                  # Pre-training pipeline (Muon+AdamW, packing, chunked CE)
 ├── optimizers.py             # Hybrid Muon + AdamW optimizer implementation
 ├── tokenizer.py              # EOS 50k BPE Tokenizer wrapper
 ├── generate_text.py          # Inference & speculative decoding script
 ├── build_large_dataset.py    # Multi-domain dataset generator
 └── optim/                    # High-performance Triton & PyTorch kernel modules
      ├── kernels.py           # SRAM-persistent fused recurrent kernels
      ├── grouped_gemm.py      # Grouped GEMM batched MoE execution
      ├── fused_swiglu.py      # Torch.compile fused SwiGLU activation
      ├── quantized_kv.py      # FP8/INT8 quantized KV cache manager
      ├── mup.py               # Model Upstream Parallelization (mUP) scaling
      ├── advanced_loops.py    # Cross-Loop Attention & Loop-Bypass Router
      ├── ring_mla.py          # Distributed Ring Multi-Latent Attention
      └── checkpoint.py        # Selective submodule gradient checkpointing
```

---

## ⚡ Quickstart

### Installation
```bash
git clone https://github.com/your-username/EOS-LLM.git
cd EOS-LLM
pip install torch transformers accelerate
```

### Pre-Training Command
```bash
python train.py \
  --dataset_path eos_v2_corpus.txt \
  --output_dir ./checkpoints/eos_run \
  --seq_len 2048 \
  --batch_size 4 \
  --max_steps 50000 \
  --save_steps 1000 \
  --optimizer muon
```

### Text Generation / Inference
```bash
python generate_text.py \
  --checkpoint_dir ./checkpoints/eos_run/checkpoint-50000 \
  --prompt "Solve step-by-step: 15 + 27 = " \
  --max_new_tokens 64
```

---

## 🔒 License & Intellectual Property

Copyright © 2026. All rights reserved.  
*This codebase is proprietary software. Unauthorized copying, distribution, or commercial modification is strictly prohibited.*
