# 🌌 EOS & Rhen ("The Brain"): Architectural Blueprint & Advanced Foundations

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Architecture: Recurrent MoE + Rhen](https://img.shields.io/badge/Architecture-Recurrent%20MoE%20%2B%20Rhen-blueviolet.svg)]()
[![Context: 8k Native / 16k YaRN](https://img.shields.io/badge/Context-8k%20Native%20%2F%2016k%20YaRN-success.svg)]()
[![Parameters: 230.73M](https://img.shields.io/badge/Parameters-230.73M%20Trainable-blue.svg)]()

> **EOS** is an open-source neural cognitive architecture combining **parameter-shared recurrent depth**, **Shared-Expert Fine-Grained Sparse Mixture-of-Experts (SE-MoE)**, **Differential Multi-Latent Attention (Diff-MLA)**, **Multi-Token Prediction (MTP)**, **State-Space Memory Highway (SSM Bridge)**, and the **Rhen Latent Decision Engine ("The Brain")** — a non-autoregressive "System One" decision engine.

---

## 🏛️ Comprehensive Architectural Blueprint

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

## 🚀 Architectural Foundations & Innovations

### 1. Differential Multi-Latent Attention (Diff-MLA)
Cancels common-mode background noise and sharpens signal-to-noise ratio over dense state prefixes:
$$A_1 = \text{Softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d_k}}\right), \quad A_2 = \text{Softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d_k}}\right)$$
$$\text{DiffAttn}(X) = (A_1 - \lambda \odot A_2) V$$
$$\lambda = \exp(\lambda_{q_1} \cdot \lambda_{k_1}) - \exp(\lambda_{q_2} \cdot \lambda_{k_2}) + \lambda_{\text{init}}$$

### 2. Shared-Expert Mixture-of-Experts (SE-MoE)
Hybrid expert architecture combining $N_{\text{shared}} = 1$ dedicated universal expert with Top-$K$ routed specialist experts ($N_{\text{routed}} = 8, K = 2$):
$$\text{MoE}(x) = \text{Expert}_{\text{shared}}(x) + \sum_{i \in \text{Top-}K} g_i(x) \cdot \text{Expert}_i(x)$$
- Uses **auxiliary-loss-free dynamic load balancing** via learnable expert routing biases updated per batch.

### 3. State-Space Memory Highway (SSM Bridge)
Discretized continuous-time recurrent highway bridging sequential recurrent loop passes:
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t$$
$$\bar{A} = \exp(-\Delta \cdot \text{Softplus}(A_{\log})), \quad \bar{B} = \Delta \cdot B$$
- Bridges multi-loop representations and preserves long-range state dependencies without gradient vanishing.

---

## 🧠 The 7 Rhen Decision Primitives

1. **Noul (Binary Semantic Truth)**: Evaluates proposition truth with $P \in [0, 1]$.
2. **Choice (Categorical Selection)**: Selects 1 winner from $K$ candidates with strictly proper probabilities.
3. **Score (Ordered Spectrum)**: Computes expected level value over discrete ordered options using Ranked Probability Score (RPS) calibration.
4. **Cascade (Chained DAGs)**: Sequential DAG execution with counterfactual verification.
5. **MultiChoice (Subset Selection)**: Evaluates independent sigmoid selection thresholds predicted from query entropy.
6. **Range (Interval Regression)**: Continuous scalar estimation via Beta distribution shape parameters $(\alpha, \beta)$ with 95% confidence bounds.
7. **Extract (Span Pointer)**: Extracts token spans directly from state prefix representation using dual start/end pointers.

---

## 📂 Repository Structural Layout

- `config.py` — Models system hyperparameters via `EOSConfig`.
- `model.py` — Implements `EOSModel` (Recurrent Transformer, Diff-MLA, SE-MoE, SSM Bridge).
- `rhen.py` — Contains `AdvancedRhenEngine`, factorized multi-head pointer logic, and core decision primitives.
- `rhen_chunker.py` — State chunking and semantic boundary management for long context states.
- `rhen_cognition.py` — Integrates RLCD policy optimization, TDT duration thresholding, and multilingual routing.
- `rhen_pillars.py` — Handles out-of-band audit logging, geometry-based abstention, and credit assignment.
- `unified_orchestrator.py` — Bridges context building, decision routing, and cryptographic Action Card staging.

---

## 🔒 License

Released under the [MIT License](LICENSE).