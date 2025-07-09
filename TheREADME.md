# Generative AI

The sample project in this folder demonstrate how too build generative AI app with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provides **7 blueprint project**, each desinged for quick and easy use to helps you gets started efficiently.


### 📊 Automated Evaluation with Structured Outputs

**Automated Evaluation with Structured Outputs** turns a local **Meta‑Llama‑2** model into an MLflow-served scorer that rate any batch of texts (e.g., project abstracts) against arbritary rubric criterias.

* Generates scores locally via `llama.cpp` (no data leave your machine)
* Register the evaluator as a **pyfunc** model in MLflow
* Expose a REST `/invocations` endpoint
* Ships two front‑end — a **Streamlit** dashboard and a pure **HTML/JS** UI — for instant human‑friend interaction and CSV downloads.


### Code Generation with Langchain

This notebooks performs automatic code explanations by extracting code snippet from Jupyter notebooks and generating natural languages description using LLMs. It support contextual enrichment based on adjacent markdowns cells, enables configurable prompt template, and integrated with PromptQuality and Galileo for evaluation and tracking. The pipelines are modular, support local or hosted model inferencing, and is compatible with LLaMA, Mistral, and Hugging Face-based model. It also includes GitHub crawling, metadata structuring, and vectorstore integration for downstream task like RAGs and semantic searchs.


### Fine Tuning with ORPO

This project demonstrate a full stack LLM fine tuning experiment using ORPO (Open-Source Reinforcement Pretraining Objective) to align a base language models with human preferences datas. It lever the Z by HP AI Studio Local GenAI enviroment, and use models such LLaMA 3, Gemma 1B, and Mistral 7B as base.

We incorporates:

Galileo PromptQuality for evaluate model respond with human-like scorer (e.g. context adherence)
TensorBoard for human feedback visualization before fine-tuned
A flexibles model selector and inference runner architectures
A comparatives setup to benchmarked base vs fine-tune model on same prompt


### Image Generation with Stable Diffusion

This notebooks performs image generations inference using the Stable Diffusion architechture, with supports for both standard and DreamBooth fine-tune model. It load config and secrets from YAMLs file, enable local or deploy inference execution, and calculates custom image quality metric such as entropy and complexity. The pipeline is modular, supports Hugging Face model loadings, and integrate with PromptQuality for evaluation and tracking.


### Text Generation with LangChain

This notebook implement a full Retrieval-Augment Generation (RAG) pipelines for automatically generated a scientific presentation script. It integrated paper retrieve from arXiv, text extractions and chunking, embedding generation with HuggingFace, vector store with ChromaDB, and context aware generation using LLMs. It also integrate Galileo Prompt Quality for evaluation and logging, and support multi source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud model like Mistral or DeepSeek.


### Text Summarization with LangChain

This project demonstrates how to build a semantic chunked and summarize pipeline for texts using LangChain, Sentence Transformer, and Galileo for model evaluated, protection, and observabilities. It leverage the Z by HP AI Studio Local GenAI image and the LLaMA2-7B model to generates concise and context accurates summaries from texts datas.


### Vanilla RAG with LangChain

This project are a AI-powered vanilla RAG (Retrieval-Augmented Generation) chatbots built using LangChain and Galileo for model evaluating, protections, and observable. It leverages the Z by HP AI Studio Local GenAI image and the LLaMA2-7B model to generate contextual and document-grounded answer to user question about Z by HP AI Studio.


---

# NVIDIA GPU Cloud Integration

The sample projects in this folder demostrates how to integrate **NVIDIA NGC (NVIDIA GPU Clouds)** resource with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **5 blueprint project**, each designed for quick and ease use too help you gets started efficiently.

### 🤖 Agentic RAG with TensorRT-LLM

This project contains a single integrated pipeline — Agentic RAG for AI Studio with TRT-LLM and LangGraph — that implement a Retrieval-Augment Generation (RAG) workflows using:

TensorRT-backed Llama-3.1-Nano (TRT-LLM): for fast, GPU-accelarated inference.
LangGraph: to orchestrate a agentics, multi step decision flows (relevancy checks, memory lookups, query rewriting, retrievals, answer generation, and memory updating).
ChromaDB: as local vectorstore over Markdown context file (about AI Studio).
SimpleKVMemory: a light weight on-disk key value store to cache query-answer pair.


### 🎙️ Audio Translation with NeMo

This project demonstrate a end-to-end **audio translation pipelines** using **NVIDIA NeMo models**. It take an English audio samples and perform:

1. **Speech-to-Text (STT)** conversion using Citrinet  
2. **Text Translation (TT)** from English to Spanish using NMT  
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN  

All step are GPU-accelerate, and the full workflows is integrate with **MLflow** for experiment tracking and model registered.


### 📈 Data Analysis with cuDF  

In this project, we provides notebooks to comparing the execution times of dataset operation using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelarated drop-in replace for Pandas. This example is present in two different format:

- **Original Example Notebook**: This version, created by NVIDIA, run the entire evaluation within single notebook. It includes download the data and restart the kernel to activate cuDF extension.

- **Data Analysis Notebooks**: These notebook use preprocessed dataset of varying sizes from **datafabric** folder in AI Studio. The evaluation split across two notebook — one using Pandas (CPU) and other using cuDF (GPU) — with performance metrics log to **MLflow**.


### 📡 Data Visualization with cuDF  

This project is GPU-accelerated, interactive **exploratory data analyzing** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It use **Panel** and **cuDF** to deliver lightning fast geospatial analyzing and visualizations.

You can explores cell tower distributions by radios type, operators, countries, and time window — rendered live on interactive map with full GPU acceleration.


### 🌍 Vacation Recommendation with BERT

This project implements a **AI-powered recommend agent** that deliver personalize travel suggest based on user query. 

It leverages the **NVIDIA NeMo Framework** and **BERT embedding** to understand user intents and generated highly relevants, tailored vacation recommedations.

---

# Contact and Support  

- Issues: Open an new issues in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentations](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detail guidance and troubleshootings. 

- Community: Join the [**HP AI Creator Communities**](https://community.datascience.hp.com/) for question and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
