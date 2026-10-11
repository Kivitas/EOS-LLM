# 🌌 EOS & Rhen ("The Brain") Decision Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Architecture: Recurrent MoE + Rhen](https://img.shields.io/badge/Architecture-Recurrent%20MoE%20%2B%20Rhen-blueviolet.svg)]()
[![Context: 8k Native / 16k YaRN](https://img.shields.io/badge/Context-8k%20Native%20%2F%2016k%20YaRN-success.svg)]()
[![Parameters: 230.73M](https://img.shields.io/badge/Parameters-230.73M%20Trainable-blue.svg)]()
[![Tests: 14 Suites Passing](https://img.shields.io/badge/Tests-14%20Suites%20Passing%20(100%25)-brightgreen.svg)]()
[![CI: Automated GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)]()
[![Architecture & Plan](https://img.shields.io/badge/Plan-EOS%20%2B%20Rhen%20Master%20Plan-purple.svg)](Architecture.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **EOS** is an open-source neural cognitive architecture combining **parameter-shared recurrent depth**, **Shared-Expert Fine-Grained Sparse Mixture-of-Experts (SE-MoE)**, **Differential Multi-Latent Attention (Diff-MLA)**, **Multi-Token Prediction (MTP)**, **State-Space Memory Highway (SSM Bridge)**, and the **Rhen Latent Decision Engine ("The Brain")** — a non-autoregressive "System One" decision engine.
>
> ⚡ **Full architecture, benchmarks, and roadmap**: see [Architecture.md](Architecture.md) for full deep-dive mathematical blueprints.
>
> 🧠 **Decision Engine Capabilities**: Evaluates $M$ concurrent questions over a single state prefix in $O(1)$ state compute with zero generation token overhead, finite-sample conformal prediction guarantees, cryptographic HMAC-SHA256 Action Cards, and dual-custody consensus execution.

> **Naming:** EOS was previously published under legacy names like `EOSLoopedLLM` and `LoopedLLMConfig`. The canonical symbols are now `EOSModel` and `EOSConfig` (`model_type = "eos"`), with legacy aliases preserved for backward compatibility.

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

## 🎯 The 7 Typed Semantic Decision Primitives

Rhen exposes seven typed semantic decision primitives evaluated non-autoregressively in continuous latent space without text generation overhead:

1. **Noul (Binary Semantic Truth)**: Evaluates proposition truth returning $P(\text{True}) \in [0, 1]$ and confidence metric $c = |2p - 1|$.
2. **Choice (Categorical Selection)**: Selects winning option from $K$ candidates with strictly proper normalized probabilities ($\sum p_k = 1.0$).
3. **Score (Ordered Spectrum)**: Positions state along discrete scale levels mapped to continuous expected value via Ranked Probability Score (RPS).
4. **Cascade (Chained DAGs)**: Sequential multi-stage conditional decision pipelines.
5. **MultiChoice (Subset Selection)**: Evaluates independent sigmoid candidate probabilities with query-adaptive entropy thresholds.
6. **Range (Continuous Interval)**: Estimates scalar values on $[y_{\min}, y_{\max}]$ using predicted Beta distribution parameters $(\alpha, \beta)$ with 95% confidence bounds.
7. **Extract (Span Pointer)**: Extracts text values (IPs, UUIDs, hostnames) directly from state text via dual start/end bilinear pointers.

---

## ⚡ Inter-Layer Communication Bridges

1. **Input Layer $\leftrightarrow$ RHEN (Delta Caching & Depth Embeddings)**: `<|state_diff|>` token provides $5-10\times$ token speedup on continuous state updates. Learned depth embeddings map structural indentation ($0 \dots 15$).
2. **Recurrent Core $\leftrightarrow$ RHEN (Channel B+ & Channel D+)**: Multi-slot layer steering targets loop-specific subspaces, while Modern Hopfield associative memory stores trajectory states to prevent recurrent drift.
3. **RHEN $\leftrightarrow$ Output Layer (Dual-Path Conditioning)**: Non-autoregressive decision latents directly cross-condition downstream token generation heads, locking text explanations to latent decision outputs.

---

## 🚀 Quick Start & Usage

```python
from config import EOSConfig
from model import EOSModel
from rhen import Choice, Noul, MultiChoice, Range, Extract
from unified_orchestrator import UnifiedAIOrchestrator

# 1. Initialize config and model
config = EOSConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    max_recurrent_loops=3,
    use_diff_mla=True,
    use_aux_free_balancing=True,
)
model = EOSModel(config)
orchestrator = UnifiedAIOrchestrator(config)

# 2. Build structured context
ctx = orchestrator.create_context_builder()
ctx.add_entity("Cluster", "prod-us-east-1", {"nodes": 48, "status": "degraded"})
ctx.add_json({"p99_latency_ms": 840, "error_rate": 0.048})

# 3. Execute decisions with conformal sets and cryptographic action cards
result = orchestrator.unified_decide_sync(
    state=ctx,
    questions=[
        Choice("Primary action", ["Scale Pods", "Throttle API", "Failover Region"]),
        Range("Estimated recovery time", min_val=0.0, max_val=120.0),
        MultiChoice("Active incident flags", ["HighLatency", "RateLimitBreached"]),
        Extract("cluster_id", query="Target cluster identifier"),
    ],
    use_three_layer=True,
    auto_stage_actions=True,
)

# 4. Process calibrated outcomes and HMAC-signed cards
for ans in result["final_answers"]:
    print(f"Decision: {ans.get('outcome')} (Confidence: {ans.get('confidence'):.2%})")

for card in result["staged_actions"]:
    print(card.to_markdown())
```

---

## 🧪 Master Test Runner

Execute the automated test runner across all 14 test suites and architectural gates:

```bash
python run_tests.py              # Run complete test suite (14 test suites)
python run_tests.py --fast       # Fast-path verification loop (<15s)
python run_tests.py --slow       # Execute heavy architectural gates
python run_tests.py --list       # Display test matrix and suite classifications
```

---

## 📄 License

EOS & the Rhen Decision Engine are released under the [MIT License](LICENSE).