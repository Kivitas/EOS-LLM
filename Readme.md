# 🌌 EOS & Rhen ("The Brain") Decision Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Architecture: Recurrent MoE + Rhen](https://img.shields.io/badge/Architecture-Recurrent%20MoE%20%2B%20Rhen-blueviolet.svg)]()
[![Context: 8k Native / 16k YaRN](https://img.shields.io/badge/Context-8k%20Native%20%2F%2016k%20YaRN-success.svg)]()
[![Parameters: 230.73M](https://img.shields.io/badge/Parameters-230.73M%20Trainable-blue.svg)]()
[![Tests: 14 Suites Passing](https://img.shields.io/badge/Tests-14%20Suites%20Passing%20(100%25)-brightgreen.svg)]()
[![CI: Automated GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **EOS** is an open-source neural cognitive architecture combining **parameter-shared recurrent depth**, **Shared-Expert Fine-Grained Sparse Mixture-of-Experts (SE-MoE)**, **Differential Multi-Latent Attention (Diff-MLA)**, **Multi-Token Prediction (MTP)**, **State-Space Memory Highway (SSM Bridge)**, and the **Rhen Latent Decision Engine ("The Brain")** — a non-autoregressive "System One" decision engine.
>
> 🧠 **Core Capabilities:** Evaluates $M$ concurrent questions over a single state prefix in $O(1)$ state compute with zero token generation overhead, finite-sample conformal prediction guarantees, cryptographic HMAC-SHA256 Action Cards, and dual-custody consensus execution.
>
> 📖 **Full Architectural Deep-Dive:** Check out the complete mathematical and theoretical specification in [Architecture.md](Architecture.md).

> **Naming Convention Note:** EOS was originally developed under legacy nomenclature (`EOSLoopedLLM`, `LoopedLLMConfig`). The canonical symbols are now `EOSModel` and `EOSConfig` (`model_type = "eos"`), with legacy aliases fully supported for backward compatibility across all modules.

---

## 🏛️ High-Level System Overview & Conceptual Architecture

```mermaid
flowchart TD
    subgraph Context_Ingestion ["1. Context Ingestion & Incremental Caching"]
        RAW["Raw State: JSON / YAML / CSV / Audio / Vision"]
        DIFF{"State Changed?"}
        RAW --> DIFF
        DIFF -- "Incremental Delta" --> S_DIFF["<|state_diff|> Delta Caching (5-10x Speedup)"]
        DIFF -- "Fresh Sequence" --> S_FULL["<|state|> Full Packaging + Depth Embeddings e_depth (0..15)"]
        S_DIFF --> TOK["EOSTokenizer with Unforgeable Delimiters"]
        S_FULL --> TOK
    end

    subgraph Input_Layer ["2. Input Layer & Hypothesis Bank Priming"]
        TOK --> EMB["Token Embeddings + YaRN RoPE (8k Native / 16k Extended)"]
        EMB --> PRE["Pre-Loop Input Layer -> H_shallow"]
        PRE --> PRIME["Slot Priming: D_slots^(0) = D_base + slot_role_embeds + Pool(H_0) W_prime"]
    end

    subgraph Recurrent_Core ["3. Recurrent Core (Loops 1..12) & Bidirectional Bridges"]
        direction TB
        SSM["State-Space Memory Highway (m_l = σ(W_A) m_(l-1) + tanh(W_B H_l))"]
        DIFF_MLA["Differential Multi-Latent Attention (Diff-MLA: A_1 - λ A_2)"]
        SE_MOE["Shared-Expert MoE (1 Shared + 8 Routed, Aux-Loss-Free Bias)"]
        
        SSM --> DIFF_MLA --> SE_MOE
        
        subgraph RHEN_Recurrent_Bridge ["RHEN Bidirectional Loop Coupling"]
            HOPFIELD["Channel D+: Modern Hopfield Associative Memory (Trajectory Recall)"]
            SLOT_STEER["Channel B+: Multi-Slot Layer Steering (Slot i targets Loop i subspace)"]
            KV_GATE["Channel C+: Multiplicative KV Gating (g_KV = σ(W_kv Z))"]
            ACT_BOOST["Channel E: Instant Certainty Feedback to ACT Halting Unit"]
        end

        SE_MOE -- "H_l" --> HOPFIELD
        PRIME -.-> HOPFIELD
        HOPFIELD --> SLOT_STEER & KV_GATE & ACT_BOOST
        SLOT_STEER -- "Steering Residual" --> SE_MOE
        KV_GATE -- "Attention Gating" --> DIFF_MLA
    end

    Recurrent_Core -- "ACT Early Exit / Max Loops" --> POST["Post-Layers & RMSNorm -> H_deep"]

    subgraph The_Brain ["4. The Rhen Brain: Latent Readout & Decision Heads"]
        HMLF["Hierarchical Multi-Layer Latent Fusion (H_shallow, H_middle, H_deep)"]
        DLHP["Dynamic Latent Head Probing (DLHP): Probes compressed MLA KV latents"]
        POST --> HMLF
        DIFF_MLA -.-> DLHP
        HMLF & DLHP --> READOUT["Multi-Head Factorized Pointer Heads (H=4, rank=128)"]
        
        READOUT --> NOUL["1. Noul: P(True) ∈ [0, 1]"]
        READOUT --> CHOICE["2. Choice: 1-of-K Simplex"]
        READOUT --> SCORE["3. Score: Ordinal Spectrum (RPS)"]
        READOUT --> CASCADE["4. Cascade: Conditional DAG"]
        READOUT --> MULTICHOICE["5. MultiChoice: Powerset Subset Selection"]
        READOUT --> RANGE["6. Range: Continuous Bounded Interval [y_min, y_max]"]
        READOUT --> EXTRACT["7. Extract: Direct Pointer Span Slot-Filling"]
    end

    subgraph Calibration_Safety ["5. Calibration, Conformal Sets & Safety"]
        CONFORMAL["Finite-Sample Conformal Sets: C(S) guaranteed (1-α) coverage"]
        ENERGY["Helmholtz Free-Energy OOD: E(q, S) = -τ log ∑ exp(s/τ)"]
        UNCERT["Epistemic vs. Aleatoric Uncertainty Decomposition"]
        
        The_Brain --> CONFORMAL & ENERGY & UNCERT
        
        CONFORMAL --> ROUTER{"Self-Conscious Router"}
        ROUTER -- "Singleton |C(S)|=1 & High Conf" --> RES_FAST["Layer 1: Fast Semantic Execution"]
        ROUTER -- "Ambiguous |C(S)| > 1" --> REASON["Layer 2: Deliberative Reasoning (Counterfactual Verification)"]
        ROUTER -- "High Energy E > 4.5 / OOD" --> GEN["Layer 3: Creative Synthesis (Structured Explanation)"]
        
        RES_FAST & REASON & GEN --> ACTION_STAGING["Action Staging Engine (HMAC-SHA256 Signed Cards)"]
        ACTION_STAGING --> AUDIT["Audited Execution (Low / Medium / High / Critical with Dual-Custody)"]
    end
```

---

## 🎯 The 7 Typed Semantic Decision Primitives

Rhen exposes seven typed semantic decision primitives evaluated non-autoregressively in continuous latent space:
1. **Noul:** Evaluates binary proposition truth ($P(\text{True}) \in [0, 1]$)