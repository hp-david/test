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
