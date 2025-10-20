# 🧠 Controllable Question Generation  
### *Phased Training for Controllable Question Generation: A Bloom’s Taxonomy–Aligned Approach*

---

## 📘 Overview  
This repository contains the implementation of a **phased-training based Question Generation (QG) system** designed for **educational applications**.  
The model is capable of generating **high-quality, Bloom’s Taxonomy–aligned questions** with controllable difficulty.  

This approach introduces a **three-stage curriculum learning pipeline** that progresses from simple factual datasets to complex reasoning and Bloom-level cognitive question generation.  
By combining **curriculum learning**, **token-based control**, and **multi-head model architectures**, it establishes a foundation for **AI-driven educational content generation**.

---

## 🎯 Key Objectives  
- Develop an **educationally deployable Question Generation model** with difficulty control.  
- Employ a **phased training curriculum** to enhance generalization and stability.  
- Align question complexity with **Bloom’s Taxonomy** levels:  
  *Remembering, Understanding, Applying,* and *Analyzing*.  
- Evaluate using comprehensive **lexical and semantic metrics**.

---

## 🧩 System Architecture  

### **1️⃣ Phased Training Framework**
- **Phase 1:** Factual QG using *SQuAD*, *MCTest*, *FairytaleQA*.  
- **Phase 2:** Reasoning and multi-hop QG using *RACE*, *HotpotQA*, *CosmosQA*.  
- **Phase 3:** Bloom-level cognitive control using *EduQG* and *Gemini 2.5 Flash*–generated datasets.  

A progressive **8:2 data reuse ratio** was applied between phases to prevent catastrophic forgetting and promote knowledge transfer.

### **2️⃣ Control Mechanisms**
- **Input-Level Conditioning:** Introduced special tokens representing Bloom levels (e.g., `<REM>`, `<UND>`, `<APP>`, `<ANA>`).  
- **Multi-Head Decoder Architecture:** Added four decoder heads, each dedicated to a specific cognitive level.  

### **3️⃣ Optimization & Training**
- Implemented using **Hugging Face Transformers’ Seq2SeqTrainer**.  
- **Bayesian optimization** with **Optuna (TPE sampler)** for hyperparameter tuning.  
- Objective: Minimize **Negative Log-Likelihood (NLL)** loss over target sequences.

---

## 🧠 Models Implemented  
| Model | Parameters | Purpose |
|-------|-------------|----------|
| **BART (Base, Small, Large)** | 86M–400M | Main backbone for controllable question generation |
| **Distilled BERT** | 66M | Lightweight baseline |
| **T5-Base** | 220M | Text-to-text benchmark model |

---

## 📚 Dataset Summary  

| **Phase** | **Datasets Used** | **Samples** | **Type** |
|------------|------------------|--------------|-----------|
| **1** | SQuAD, MCTest, FairytaleQA | 111K | Factual / Narrative |
| **2** | RACE, HotpotQA, CosmosQA | 100K | Reasoning / Multi-hop |
| **3** | EduQG, Gemini 2.5 Flash | 18K | Bloom-level Cognitive |

> Data integrity ensured via 8:1:1 splits and strict prevention of data leakage between phases.

---

## 🧮 Evaluation Metrics  
Performance evaluated using **lexical, syntactic, and semantic** measures:  
- **BLEU-1** – N-gram fluency  
- **ROUGE-L** – Structural similarity  
- **METEOR** – Semantic alignment  
- **BERTScore** – Contextual similarity  
- **Self-BLEU**, **Distinct-1/2** – Diversity and redundancy indicators  

---

## 🧾 Results Summary  

| Model | BLEU-1 | ROUGE-L | METEOR | BERTScore | Distinct-1 |
|--------|---------|----------|----------|-------------|-------------|
| **BART-Base** | **0.64** | **0.43** | **0.57** | **0.89** | **0.58** |
| BART-Large | 0.54 | 0.24 | 0.58 | 0.88 | 0.57 |
| T5-Base | 0.35 | 0.29 | 0.29 | 0.90 | 0.59 |
| Distilled-BERT | 0.11 | 0.15 | 0.26 | 0.86 | 0.07 |

### Bloom-Level Breakdown (BART-Base)
| Level | BLEU-1 | ROUGE-L | METEOR | BERTScore |
|--------|--------|----------|----------|------------|
| Remembering | 0.67 | 0.45 | 0.58 | 0.89 |
| Understanding | 0.64 | 0.41 | 0.56 | 0.89 |
| Applying | 0.66 | 0.44 | 0.58 | 0.89 |
| Analyzing | 0.67 | 0.44 | 0.58 | 0.89 |

---

## 🔍 Explainable AI Integration  
Explainability was introduced using **BertViz**, which visualized **encoder-decoder cross-attention** patterns.  
These visualizations confirmed that the model’s attention corresponded meaningfully to contextual and cognitive relevance, strengthening model interpretability.

---

## 🧪 Ablation & Efficiency Studies  
- Conducted ablation studies to analyze **phase effects**, **quantization**, and **sampling variations**.  
- Applied **Post-Training Quantization (PTQ)** (16-bit, 8-bit, 4-bit) for compression with minimal accuracy drop.  
- **BART-Base** achieved the best balance of accuracy, efficiency, and latency (0.15s).

---

## 🚀 Key Contributions  
- Introduced a **three-phase curriculum learning approach** for QG.  
- Achieved **granular control over question difficulty** following Bloom’s Taxonomy.  
- Created a **custom Bloom-level dataset** via **Gemini 2.5 Flash** prompt engineering.  
- Implemented **Explainable AI (BertViz)** for transparency in question generation.  
- Delivered **state-of-the-art results**, outperforming prior QG models on BLEU and ROUGE metrics.

---

## 🧩 Technologies Used  
- **Python 3**  
- **PyTorch**  
- **Hugging Face Transformers**  
- **Optuna** (Bayesian Optimization)  
- **BertViz** (Attention Visualization)  
- **Pandas**, **NumPy**, **Matplotlib**

---

## 🏁 Conclusion  
This project demonstrates a **phased-learning approach** that allows Question Generation models to produce contextually relevant and cognitively controlled educational questions.  
The **BART-Base model**, enhanced with **token-based conditioning**, provides a robust and interpretable framework for **adaptive learning and AI-assisted assessment systems**.

---

