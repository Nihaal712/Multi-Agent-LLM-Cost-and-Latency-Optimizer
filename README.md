# LLM Cost & Latency Optimizer

A production-oriented **GenAI system design project** that dynamically optimizes **LLM cost, latency, and quality** using **policy-driven decision making**, **deterministic rules**, and **full explainability**.

This project demonstrates how real-world LLM systems are built beyond simple “prompt → model → response” pipelines.

---

## 🚀 Project Motivation

In most GenAI applications:

- The **same LLM** is used for every query
- The **same context size** is always sent
- The **same retrieval depth** is used

This leads to:
- Unnecessary **cost**
- Higher **latency**
- Poor **resource utilization**

### This project solves that.

It dynamically decides:
- Which **model tier** to use (SMALL / MEDIUM / LARGE)
- How much **context** to send
- How many documents to **retrieve**

Based on:
- Query complexity
- Optional **budget constraint**
- Optional **SLA (speed requirement)**

All decisions are:
- **Deterministic**
- **Explainable**
- **Traceable**
- **Testable**

---

## 🧠 Core Design Principles

1. **LLMs are used only for judgment, not decisions**
   - Classification and reasoning → LLM
   - Enforcement and control → deterministic code

2. **Policy-driven architecture**
   - Cost policy
   - Latency policy
   - Conflict resolution rules

3. **Full explainability**
   - Every step produces a structured JSON trace
   - Trace is returned to the UI

4. **Deterministic execution**
   - No randomness in core logic
   - Temperature = 0
   - Same input → same output

5. **Async-first**
   - Parallel policy evaluation
   - Non-blocking LLM calls
   - Optimized latency

---

## 🏗️ High-Level Architecture
User Query
↓
Query Complexity Agent (LLM)
↓
Cost Policy Agent ──┐
├──► Execution Controller (Deterministic Rules)
Latency Policy Agent ┘
↓
Optimized RAG Pipeline
↓
Answer + Cost/Latency Estimates + Full Trace
---

## 🔧 Key Concepts Used

- **LangChain (LCEL)** for composable chains
- **LangGraph** for explicit state orchestration
- **FAISS** for vector similarity search
- **Groq (LLaMA models)** for fast LLM inference
- **Streamlit** for interactive UI
- **Pydantic** for schema validation
- **tiktoken** for token counting

---

## 📁 Project Structure
llm-cost-latency-optimizer/
│
├── app.py                 # Streamlit UI
├── main.py                # LangGraph orchestration + CLI
├── config.py              # Policies, tiers, defaults, conflict rules
├── agents.py              # LLM-based policy agents (JSON outputs)
├── core.py                # Deterministic controller + RAG logic
├── utils.py               # Helpers (LLM calls, tracing, token counting)
├── requirements.txt
├── .env.example
└── README.md
---

## 🤖 Model Tiers

Instead of hardcoding models, the system uses **tiers**:

| Tier | Characteristics |
|----|----|
| SMALL | Low cost, low latency |
| MEDIUM | Balanced |
| LARGE | Higher cost, better reasoning |

Actual models are mapped via a registry, making the system **model-agnostic**.

---

## 📊 Decision Factors

### 1. Query Complexity
LLM-based classification:
- `simple`
- `medium`
- `complex`

Used only for **judgment**, not control.

---

### 2. Cost Policy
Based on:
- Complexity
- Optional budget

Produces:
- Allowed model tiers
- Maximum token budget

---

### 3. Latency Policy
Based on:
- Complexity
- SLA (`fast`, `balanced`, `relaxed`)

Produces:
- Preferred tier
- Maximum retrieval depth

---

### 4. Conflict Resolution (Deterministic)
Rules resolve conflicts such as:
- SLA vs Budget
- Context limits
- Retrieval depth limits

**Example rule:**
> SLA takes precedence over budget for latency-sensitive queries.

---

## 📚 RAG Pipeline

- Documents embedded using HuggingFace embeddings
- Indexed using FAISS
- Retrieval depth dynamically controlled
- Context size capped based on policies
- Token-safe prompt construction

---

## 🔍 Explainability & Tracing

Every step produces structured trace data:

json
{
  "step": "latency_policy",
  "inputs": {"sla": "fast"},
  "outputs": {"preferred_tier": "SMALL"},
  "timestamp": "..."
}

⚙️ Setup Instructions
1. Clone repository
   git clone <repo-url>
   cd llm-cost-latency-optimizer
2. Install dependencies
   pip install -r requirements.txt
3. Environment setup
   cp .env.example .env
Add your Groq API key:
GROQ_API_KEY=your_key_here

---

▶️ Run the Project
CLI Mode : python main.py --query "Explain transformers" --sla fast
Streamlit UI : streamlit run app.py

---

🎯 What This Project Demonstrates
	•	Real-world GenAI system design
	•	Policy-based LLM optimization
	•	LangGraph orchestration
	•	Deterministic decision making
	•	Explainability-first architecture

This project is intentionally designed to reflect how production GenAI systems are built, not just demo-level applications.

---

📌 Author Notes

Built as a learning-focused project to deeply understand:
	•	LLM cost/latency trade-offs
	•	Async GenAI pipelines
	•	Explainable AI system design
