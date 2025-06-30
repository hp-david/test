<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Projects for HP AI Studio 🚀 </h1>

# Content  
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Science](#data-science)
- [Deep Learning](#deep-learning)
- [Generative AI](#generative-ai)
- [NVIDIA GPU Cloud (NGC) Integration](#nvidia-gpu-cloud-integration)
- [Contact and Support](#contact-and-support)

---

# Overview 

This repository have a collections of example projects that can be runned fast and effortless, and is design for integrate good with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html). Each project runs end-to-end and offers out-of-box and easy-to-use solution for many domain like datascience, ML, deep-learnings and Gen AI.  

The projects is using some local open source model like **LLaMA** (Meta), **BERT** (Google), and **CitriNet** (NVIDIA), also few hosted models from **Hugging Face**. Those example include case like **data visual**, **stock predict**, **voice to text**, **agent rag**, and others more.  

We is always adding new projects. If you got idea or want to see something add to [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html), just open new issues on repo — we like your feedbacks!

---

# Repository Structure 

- data-science  
  - classification-with-svm  
  - data-analysis-with-var  
- deep-learning  
  - classification-with-keras  
  - question-answering-with-bert  
  - recommendation-system-with-tensorflow  
  - spam-detection-with-nlp  
  - super-resolution-with-fsrcnn  
  - text-generation-with-rnn  
- generative-ai  
  - automated-evaluation-with-structured-outputs  
  - code-generation-with-langchain  
  - fine-tuning-with-orpo  
  - image-generation-with-stablediffusion  
  - text-generation-with-langchain  
  - text-summarization-with-langchain  
  - vanilla-rag-with-langchain  
- ngc-integration  
  - agentic-rag-with-tensorrtllm  
  - audio-translation-with-nemo  
  - data-analysis-with-cudf  
  - data-visualization-with-cudf  
  - vacation-recommendation-with-bert  

---

# Data Science

The sample in this folder show how build datascience things with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).  

We giving 2 blueprint, each made for quick start and use easily.  

### 🌸 Classification with SVM

This project is simple **classification** task about predicting kind of **iris flowers**.  

It run on **Data Science Workspace**, and show basic supervised model for classifying things.

### 🏙️ Data Analysis with VAR

This one doing **regression** experiment with **covid mobility** dataset.  

It tells how cities movement changes during pandemic. Run on **Data Science Workspace** too.

---

# Deep Learning

These project showing how to use deep learning with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).  

We give 6 blueprints, so you can start fast.

### 🖌️ Classification with Keras

It do simple **image classification** using **TensorFlow** framework.  

Model train to tell which digit from **MNIST**, run in **Deep Learning Workspace**.

### 🧠 Question Answering with BERT

This shows **BERT QA** usecase. It train model to answer stuff and also can use **Hugging Face** pretrained.  

It use **MLflow** for deploy so it can infer via API.

### 🎬 Recommendation System with TensorFlow

This build basic **movie recommender** with **TensorFlow**.  

Model learn what user might like. Needs **Deep Learning Workspace** to train.

### 🚫 Spam Detection with NLP

This is a **text classifier** for detect **spam messages**.  

Use deep learning model. Require **Deep Learning Workspace** to run well.

### 🖼️ Super Resolution with FSRCNN

Computer Vision example that improves image resolutions.  

Use CNNs to make picture better look.

### ✍️ Text Generation with RNN

Make text generation that predict **char by char**.  

It train on Shakespeare dataset and output character sequences.

---

# Generative AI

This folder shows how to do GenAI with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).  

We got 7 blueprints for start fast.

### 📊 Automated Evaluation with Structured Outputs

Turn a **Llama 2** into MLflow serve scorer that rate batches texts.  

* Score done local using llama.cpp  
* Evaluator is registered as **pyfunc** in MLflow  
* REST `/invocations` endpoint  
* Has **Streamlit** and **HTML** UI too  

### Code Generation with Langchain

This notebook take code from Jupyter and explain it. Use LLMs to make texts explain codes.  

It support local or remote model. Can use LLaMA, Mistral, etc. Also do GitHub crawling and RAG stuff.

### Fine Tuning with ORPO

This shows full stack LLM finetune using ORPO.  

Use models like **LLaMA 3**, **Gemma**, **Mistral 7B**. Track scores with Galileo PromptQuality and TensorBoard.  

Also compare base model vs fine-tuned results on same prompt.

### Image Generation with Stable Diffusion

Make image from prompt using Stable Diffusion.  

Support DreamBooth too. Loads model with Hugging Face. Metrics like entropy is tracked.

### Text Generation with LangChain

Build full **RAG pipeline** that generate science presentation.  

Do paper retrieval, chunking, embedding, and response using LLM. Can run with local or HF-hosted model.

### Text Summarization with LangChain

Summarize text by chunking and running LLMs.  

Use LLaMA2-7B and Galileo for tracking. Work in local GenAI setup.

### Vanilla RAG with LangChain

Simple chatbot with LangChain and RAG. Use LLaMA2-7B.  

Answer question using local files. Eval with Galileo tools.

---

# NVIDIA GPU Cloud Integration

These projects showing how to use **NVIDIA NGC** things inside [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).  

We provide 5 sample that is quick for start.

### 🤖 Agentic RAG with TensorRT-LLM

Full pipeline of RAG with **TRT-LLM** and LangGraph.  

Use TensorRT for fast inference. LangGraph for agents. ChromaDB as store. SimpleKVMemory for cache.

### 🎙️ Audio Translation with NeMo

Translate English voice to Spanish.  

1. **STT** with Citrinet  
2. **Translate** to Spanish  
3. **TTS** with FastPitch + HiFiGAN  

All run with GPU and track in **MLflow**.

### 📈 Data Analysis with cuDF  

Notebook show time difference of using Pandas vs cuDF (GPU).  

Two versions — one from NVIDIA, other with AI Studio data. Logs to **MLflow**.

### 📡 Data Visualization with cuDF  

EDA dashboard using **Panel** and **cuDF**.  

Show cell tower map by operator, country, and time. All is GPU-accelerate.

### 🌍 Vacation Recommendation with BERT

Recommend travel places based on what user say.  

Use BERT from NeMo to embed queries and return best match.

---

# Contact and Support  

- Issues: Open issue on [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints)  

- Docs: Check the [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview)  

- Community: Ask on [**HP AI Creator Community**](https://community.datascience.hp.com/)  

---

> Build with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html)
