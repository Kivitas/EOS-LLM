# 🌌 EOS & Rhen ("The Brain") Latent Decision Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Architecture: Recurrent MoE + Rhen](https://img.shields.io/badge/Architecture-Recurrent%20MoE%20%2B%20Rhen-blueviolet.svg)]()
[![Context: 8k Native / 16k YaRN](https://img.shields.io/badge/Context-8k%20Native%20%2F%2016k%20YaRN-success.svg)]()
[![Parameters: 230.73M](https://img.shields.io/badge/Parameters-230.73M%20Trainable-blue.svg)]()
[![Tests: 14 Suites Passing](https://img.shields.io/badge/Tests-14%20Suites%20Passing%20(100%25)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **EOS** is an open-source neural cognitive architecture combining **parameter-shared recurrent depth**, **Shared-Expert Fine-Grained Sparse Mixture-of-Experts (SE-MoE)**, **Differential Multi-Latent Attention (Diff-MLA)**, **State-Space Memory Highways (SSM Bridge)**, and the **Rhen Latent Decision Engine ("The Brain")** — a non-autoregressive decision platform designed for continuous latent space reasoning.
>
> 🧠 **Core Capability:** Evaluates $M$ structured queries over a single ingested state in $O(1)$ token generation overhead, featuring finite-sample conformal prediction guarantees, Helmholtz free-energy anomaly detection, and HMAC-SHA256 signed action execution.
>
> 📖 **Deep-Dive Specification:** For full mathematical derivations and system architecture, see [Architecture.md](Architecture.md).

---

## 🌟 The Core Paradigm Shift: State Ingestion vs. Token Generation

Traditional transformer models rely on autoregressive token-by-token generation to solve tasks. While effective for open-ended text generation, autoregressive decoding introduces latency, high token consumption, and cumulative error propagation when applied to operational decision-making.

EOS and Rhen separate **state ingestion** from **semantic decision-making**:

1. **Recurrent State Processing:** Context sequences (telemetry, JSON, logs, codebase states) up to 16,000 tokens are processed through parameter-shared recurrent transformer layers modulated by Diff-MLA and an SSM Memory Bridge.

2. **Non-Autoregressive Probing:** Rather than generating text responses, the Rhen engine probes compressed multi-layer hidden states using factorized bilinear pointer heads.

3. **Instant Decision Payloads:** Evaluates logical classifications, value regressions, and span extractions in a single forward step without output token overhead.

---

## 🎯 The 7 Typed Semantic Decision Primitives

The Rhen Decision Engine exposes seven typed semantic decision primitives evaluated directly in continuous space:

| Primitive | Target Output | Primary Operational Use Case | 
| ----- | ----- | ----- | 
| **Noul** | $P(\text{True}) \in [0, 1]$ | Binary truth assessment & logical assertion validation | 
| **Choice** | 1-of-$K$ Simplex ($\sum p_k = 1.0$) | Discrete operational routing & branch selection | 
| **Score** | Ordinal Scale positioning | Severity rating, priority ranking & ordinal metrics | 
| **Cascade** | Execution DAG | Multi-stage conditional dependency workflows | 
| **MultiChoice** | Subsets from $K$ candidates | Multi-label identification & feature flagging | 
| **Range** | Interval $[y_{\min}, y_{\max}]$ | Continuous parameter estimation & scalar bounds | 
| **Extract** | Buffer Start/End Pointers | Sub-token span extraction & entity isolation | 

---

## 🏛️ Three-Layer Metacognitive Decision Pipeline

To maintain high precision and prevent false confidence, EOS routes queries through a three-layer execution pipeline:

```
                                  [ Incoming Input Query ]
                                             │
                                             ▼
                             [ Rhen Continuous Latent Readout ]
                                             │
                                             ▼
                             [ Conformal Set & Energy Audit ]
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
                     ▼                       ▼                       ▼
            [ Singleton C(S)=1 ]    [ Ambiguous |C(S)| > 1 ]  [ High Energy E > 4.5 ]
                     │                       │                       │
                     ▼                       ▼                       ▼
            ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
            │     Layer 1     │     │     Layer 2     │     │     Layer 3     │
            │  Fast Pathway   │     │  Deliberative   │     │ Critical Escalation│
            │  (Instant Exec) │     │ Verification Loop│    │ & Synthesis     │
            └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                     │                       │                       │
                     └───────────────────────┼───────────────────────┘
                                             │
                                             ▼
                             [ HMAC-SHA256 Signed Action Cards ]
                                             │
                                             ▼
                             [ Multi-Tier Audited Execution ]
```

1. **Layer 1 (Fast Pathway):** Executes instantly when conformal prediction sets yield a single valid option ($\vert \mathcal{C}(S) \vert = 1$) with high epistemic confidence and low free-energy ($E < 4.5$).

2. **Layer 2 (Deliberative Reasoning):** Triggered when prediction sets yield ambiguity ($\vert \mathcal{C}(S) \vert > 1$). The system initiates counterfactual verification passes across additional recurrent loops.

3. **Layer 3 (Critical Escalation & Synthesis):** Triggered when Helmholtz free-energy scores indicate out-of-distribution or corrupted inputs ($E \ge 4.5$). Automated execution halts and queries escalate to human operators or explanatory synthesis modes.

---

## 🔐 Cryptographic Action Staging & Safety Governance

Decisions generated by Rhen are packaged into secure **Cryptographic Action Cards**:

* **Payload Verification:** Encapsulates proposed target parameters, confidence margins, and conformal set bounds.

* **HMAC-SHA256 Integrity:** Cards are digitally signed at creation to prevent tampering or unauthorized modification.

* **Multi-Tier Execution:** Actions are categorized into Low, Medium, High, and Critical risk tiers. High and Critical operations require dual-custody authorization signatures before execution.

---

## 📄 License

EOS and the Rhen Decision Engine are open-source software licensed under the [MIT License](LICENSE).