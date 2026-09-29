<div align="center">

# 📄 Universal PDF RAG Chatbot
### Production-Grade, Privacy-First Conversational Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.17-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6F00?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FC521F?style=for-the-badge)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2:3b-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-22C55E?style=for-the-badge)](https://github.com/)

<p align="center">
  A local, zero-token-cost Retrieval-Augmented Generation (RAG) system capable of answering complex queries over arbitrary PDF documents. Built with LangGraph conversational checkpointing, ChromaDB vector isolation, and real-time token streaming.
</p>

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technical Stack](#-technical-stack)
- [Evaluation Benchmark](#-evaluation-benchmark)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Author & Contact](#-author--contact)

---

## 🔍 Overview
The **Universal PDF RAG Chatbot** bridges local document understanding and private conversational AI. Running completely on-device, it eliminates third-party API subscription costs and data privacy concerns. 

Unlike naive RAG pipelines, this system introduces:
- **Intelligent Query Rewriting**: Resolves conversational pronouns (*"it"*, *"their advantages"*, *"how does that work?"*) against previous conversation turns.
- **MMR Semantic Search**: Maximal Marginal Relevance retrieval prevents redundant chunks from dominating context.
- **Strict Anti-Hallucination Grounding**: Prompts force the model to answer exclusively from retrieved context or explicitly declare non-coverage.
- **Deduplicated Page Citations**: Extracts original 1-based page numbers from document metadata for instant auditability.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    classDef inputNode  fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1
    classDef procNode   fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#6B21A8
    classDef storeNode  fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#92400E
    classDef wfNode     fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D
    classDef outNode    fill:#FFE4E6,stroke:#E11D48,stroke-width:2px,color:#9F1239

    subgraph INGESTION ["Document Ingestion Pipeline"]
        A["Uploaded PDF"]:::inputNode --> B["PyPDFLoader + TOC Filter + Chapter Detection"]:::procNode
        B --> C["RecursiveCharacterTextSplitter\n1000 chars, 150 overlap"]:::procNode
        C --> D["HuggingFace Embeddings\nall-MiniLM-L6-v2"]:::procNode
        D --> E[("ChromaDB Vector Store\nIsolated by PDF Hash")]:::storeNode
    end

    subgraph WORKFLOW ["LangGraph Conversational Workflow"]
        U["User Question"]:::inputNode --> N1["rewrite_question\nResolves pronouns via history"]:::wfNode
        M[("MemorySaver\nCheckpointer")]:::storeNode --> N1
        N1 --> M
        N1 --> N2["retrieve_documents\nMMR k=8, fetch_k=24, lambda=0.5"]:::wfNode
        E -.->|Semantic Search| N2
        N2 --> N3["generate_answer\nOllama llama3.2:3b local"]:::wfNode
    end

    subgraph OUTPUT ["Streamlit Streaming UI"]
        N3 --> O1["Live Token Streaming"]:::outNode
        N3 --> O2["Collapsible Page Citations"]:::outNode
    end
```

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **📁 Arbitrary PDF Ingestion** | Ingest any document (textbooks, reports, research papers, documentation) without modifying code. |
| **⚡ Smart DB Isolation & Caching** | Each PDF is fingerprinted with a 12-character SHA-256 hash. ChromaDB collections remain isolated to prevent cross-document contamination, and re-indexing is skipped if already cached. |
| **🧠 LangGraph Multi-Turn Memory** | Checkpointer state management automatically retains context across turns, enabling fluid follow-up inquiries. |
| **🎯 Redundancy Reduction (MMR)** | Maximal Marginal Relevance balances relevance with diversity across retrieved chunks. |
| **🛑 Responsive Stop Generation** | Live token callbacks serve as active execution yield points, allowing Streamlit's **Stop** button to interrupt generation in milliseconds. |
| **📑 Verified Page Citations** | Preserves true 1-based page metadata, providing direct source verification for every generated claim. |

---

## 🛠️ Technical Stack

- **Orchestration & Workflow:** [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph + MemorySaver)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Local Large Language Model:** [Ollama](https://ollama.ai/) running `llama3.2:3b`
- **Document Processing:** `pypdf`, `langchain-text-splitters`
- **User Interface:** [Streamlit](https://streamlit.io/) with custom theme-adaptive CSS

---

## 📊 Evaluation Benchmark

> [!NOTE]
> The quantitative evaluation below was established across a comprehensive **60-question benchmark** recorded in [`evaluation/evaluation.ipynb`](evaluation/evaluation.ipynb). It serves as the baseline experimentation artifact.

<div align="center">

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Questions Evaluated** | **60** | Multi-topic comprehensive test set |
| **Retrieval Hit Rate** | **76.00%** | Relevant source page present in retrieved MMR context |
| **Average Answer Score** | **1.49 / 2.00** | Evaluated via LLM Judge (Correctness + Groundedness) |
| **Fully Correct & Grounded** | **53.33%** | Strict adherence with zero hallucinated claims |

</div>

---

## 📁 Project Structure

```text
RAG/
│
├── app.py                     # Streamlit web application & streaming UI
├── rag.py                     # Core RAG engine, vectorstore manager & LangGraph workflow
│
├── evaluation/
│   └── evaluation.ipynb       # 60-question evaluation benchmark & analysis
│
├── data/
│   └── uploads/               # Temporary runtime storage for uploaded PDFs
│
├── chroma_db/                 # Persistent vector databases isolated by document hash
│
├── requirements.txt           # Environment dependencies
├── README.md                  # Project documentation & architecture
└── .gitignore                 # Version control rules
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites (Ollama Local Model)
Ensure [Ollama](https://ollama.com/) is installed and running on your system:
```bash
ollama pull llama3.2:3b
ollama serve
```

### 2. Setup Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`, upload your PDF in the sidebar, click **Process PDF**, and start chatting!

---

## 👤 Author & Contact

**Renish Nagapara**  
AI / ML Engineer & GenAI Enthusiast

<p align="left">
  <a href="https://www.linkedin.com/in/renish-nagapara-597814329/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  &nbsp;&nbsp;
  <a href="mailto:renishpatel7606@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
</p>

*Feel free to reach out for collaborations, discussions on RAG architectures, or project feedback!*
