<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Projects for HP AI Studio 🚀 </h1>

# Content

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Science](#data-science)
- [Deep Learning](#deep-learning)
- [Generative AI](#generative-ai)
- [NVIDIA GPU Cloud (NGC) Integration](#nvidia-gpu-cloud-integration)
- [Troubleshooting](#troubleshooting) 
- [Contact and Support](#contact-and-support)

---

# Overview

This repository contains a collection of sample projects that you can run quickly and effortlessly, designed to integrate seamlessly with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html). Each project runs end-to-end, offering out-of-the-box, ready-to-use solutions across various domains, including data science, machine learning, deep learning, and generative AI.

The projects leverage local open-source models such as **LLaMA** (Meta), **BERT** (Google), and **Nemotron** (NVIDIA), alongside selected online models accessible via **Hugging Face**. These examples cover a wide range of use cases, including **data visualization**, **stock analysis**, **audio translation**, **agentic RAG applications**, and much more.

We are continuously expanding this collection with new projects. If you have suggestions or would like to see a specific sample project integrated with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html), please feel free to open a new issue in this repository — we welcome your feedback!

---

# Repository Structure

```
# Root Directory
├── data-science/                          # Projects related to classical machine learning and statistical analysis
│   ├── classification-with-svm/           # SVM-based classification implementation
│   └── data-analysis-with-var/            # Vector AutoRegression analysis workflow
│
├── deep-learning/                         # Deep learning applications using popular frameworks
│   ├── classification-with-keras/         # Image classification using Keras
│   ├── question-answering-with-bert/      # QA system built on top of BERT model
│   ├── recommendation-system-with-tensorflow/  # TensorFlow-based recommendation engine
│   ├── spam-detection-with-nlp/           # NLP-driven spam classifier
│   ├── super-resolution-with-fsrcnn/      # Image enhancement using FSRCNN
│   └── text-generation-with-rnn/          # RNN-based generative model for text
│
├── generative-ai/                         # Generative AI applications across text, code, and image
│   ├── automated-evaluation-with-structured-outputs/  # Eval pipeline for structured generation
│   ├── code-generation-with-langchain/    # Code synthesis using LangChain
│   ├── fine-tuning-with-orpo/             # ORPO-based fine-tuning procedure
│   ├── image-generation-with-stablediffusion/  # StableDiffusion-powered image generation
│   ├── text-generation-with-langchain/    # Text generation leveraging LangChain stack
│   ├── text-summarization-with-langchain/ # Summarization pipeline using LangChain
│   └── vanilla-rag-with-langchain/        # Basic Retrieval-Augmented Generation with LangChain
│
├── ngc-integration/                       # Projects leveraging NVIDIA GPU Cloud and libraries
│   ├── agentic-rag-with-tensorrtllm/      # RAG system using TensorRT-LLM and agentic planning
│   ├── audio-translation-with-nemo/       # Speech translation with NVIDIA NeMo
│   ├── data-analysis-with-cudf/           # RAPIDS cuDF-based data manipulation
│   ├── data-visualization-with-cudf/      # Visualizations using GPU-accelerated tools
│   └── vacation-recommendation-with-bert/ # Recommendation app using BERT embeddings
```

---

# Data Science

The sample projects in this folder demonstrate how to build data science applications with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **2 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

### 🌸 Classification with SVM

This project is a simple **classification** experiment focused on predicting species of **Iris flowers**.

It runs on the **Data Science Workspace**, demonstrating basic supervised learning techniques for multi-class classification tasks.

### 🏙️ Data Analysis with VAR

This project explores a **regression** experiment using **mobility data** collected during the COVID-19 pandemic.

It highlights how city-level movement patterns changed during the crisis. The experiment runs on the **Data Science Workspace**.

# NVIDIA GPU Cloud Integration

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **5 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

### 🤖 Agentic RAG with TensorRT-LLM

This project contains a single integrated pipeline—Agentic RAG for AI Studio with TRT-LLM and LangGraph—that implements a Retrieval-Augmented Generation (RAG) workflow using:

TensorRT-backed Llama-3.1-Nano (TRT-LLM): for fast, GPU-accelerated inference.
LangGraph: to orchestrate an agentic, multi-step decision flow (relevance check, memory lookup, query rewriting, retrieval, answer generation, and memory update).
ChromaDB: as a local vector store over Markdown context files (about AI Studio).
SimpleKVMemory: a lightweight on-disk key-value store to cache query-answer pairs.

### 🎙️ Audio Translation with NeMo

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet
2. **Text Translation (TT)** from English to Spanish using NMT
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.

### 📈 Data Analysis with cuDF

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Original Example Notebook**: This version, created by NVIDIA, runs the entire evaluation within a single notebook. It includes downloading the data and restarting the kernel to activate the cuDF extension.

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.

### 📡 Data Visualization with cuDF

This project is a GPU-accelerated, interactive **exploratory data analysis (EDA)** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It uses **Panel** and **cuDF** to deliver lightning-fast geospatial analysis and visualization.

You can explore cell tower distributions by radio type, operator, country, and time window — rendered live on an interactive map with full GPU acceleration.

### 🌍 Vacation Recommendation with BERT

This project implements an **AI-powered recommendation agent** that delivers personalized travel suggestions based on user queries.

It leverages the **NVIDIA NeMo Framework** and **BERT embeddings** to understand user intent and generate highly relevant, tailored vacation recommendations.

---

# Troubleshooting

This section provides solutions for common issues users may encounter when working with AI Blueprint projects in HP AI Studio:

1. **Check Hardware Compatibility**
   Each project’s README includes recommended minimum hardware specifications (e.g., RAM, VRAM). Make sure your system meets these requirements—especially when working with large models or during deployment, as insufficient resources can cause failures.

2. **Models or Datasets Not Visible After Download**
   If you download models or datasets while your workspace is running, they might not appear in the workspace. In such cases, restart your workspace to ensure they are properly recognized.

3. **Connection or SSL Errors in Notebooks**
   If you encounter SSL certificate or connection errors while accessing websites from notebooks (especially on restricted networks), verify your network settings. Consider using a proxy to bypass restrictive network constraints.

4. **File or Path Not Found Errors**
   Ensure that all required files and directories are correctly placed as specified in the project’s README. If any paths or files are missing, create or move them to the correct locations.

5. **GPU Not Available**
   For projects requiring NVIDIA GPUs, verify GPU availability by running `nvidia-smi` in the terminal. Ensure that a compatible GPU is accessible and has sufficient free memory to run the project.

6. **Deployment Errors Despite Meeting Requirements**
   Even if your hardware meets the specs, limited available RAM or VRAM can cause deployment issues. Close other running workspaces or programs to free up memory.

7. **API Timeout Issues**
   API requests triggered through user interfaces have a response timeout limit (usually 30 seconds). For long-running tasks or large inputs, use the provided notebooks instead of the UI to avoid timeout errors.

---

# Contact and Support

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
