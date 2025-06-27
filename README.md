<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Project for HP AI Studio 🚀 </h1>

# Content  
- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Data Science and machine learning](#data-science-and-machine-learning)
- [deep Learning](#deep-learning)
- [Generative ai](#generative-ai)
- [Nvidia gpu cloud](#nvidia-gpu-cloud)
- [Contact and Support](#contact-and-support)

---

# Overview 

This repo contain a collections of sample project you can run quick and effortless, made to works good with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview). Each projects runs from end-to-end, and offers out-of-box usable solution across many domain like data science, ML, DL and generative ai.  

Projects uses local open source models like **LLaMA** (meta), **BERT** (google) and **CitriNet** (nvidia), plus some online models via **Hugging Face**. You can try different usecases like **data visual**, **stocks analyse**, **audio translate**, **agent RAG** and more.  

We is adding more examples regularly. If you got suggestion or want to see something specific with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview), just open an issue in the repo — feedback are welcome!

---

# Repository structure 

- data-science-and-machine-learning
  - covid_movement_patterns_with_var
  - handwritten_digit_classification_with_keras
  - iris_flowers_classification_with_svm
  - recommender_systems_with_tensorflow
  - spam_detection_with_NLP
- deep-learning
  - bert_qa
  - super_resolution
  - text_generation
- generative-ai
  - agentic_rag_with_trt-llm_and_langgraph
  - automated_evaluation_with_structured_outputs
- ngc-integration
  - audio_translation_with_nemo_models
  - opencellid_eda_with_panel_and_cuDF
  - stock_analysis_with_pandas_and_cuDF
  - vacation_recommendation_agent_with_bert

---

# Data Science and machine learning

This samples show how to build data science and ML projects with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

There is **5 sample project** that is ready to use and help you get start fastly.

### 🌸 Iris Flower Classify with SVM

This project is basic **classify** experiment to predicts type of **Iris flower**.  

It work on **Data Science Workspace**, and show how supervised learning works for many-class problem.

### 🖌️ Handwriting Digit Classify with Keras

This model make simple **image classify** using **TensorFlow**.  

It training model for MNIST digit and needs **Deep Learning Workspace**.

### 🏙️ COVID Move Patterns with VAR

This project do **regression** using **COVID mobility dataset**.  

It looking at how city move change during pandemics. Use **Data Science Workspace**.

### 🎬 Movie Recommend System with TensorFlow

This project makes a movie **recommender system** with TensorFlow.  

It learn on user interaction and predict what movie is good. Needs **Deep Learning Workspace** to run.

### 🚫 Spam Detect using NLP

This one make **text classify** model to spot **spam message**.  

It uses deep learn model and should be runned on **Deep Learning Workspace**.

---

# deep Learning

Projects in this part shows how to build deep learning stuff using [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

There are **3 project** in here to help you get begin quickly.

### 🧠 BERT QA with MLflow

This is a simple **question answering** model using BERT. You can train or use model from **HuggingFace**.  

It use **MLflow** for inference server that answer the input text questions.

### 🖼️ Image Super Resolution with CNNs

This is **computer vision** task to enhance picture quality using CNN models.  

Train on low resolution image and produce high quality version.

### ✍️ Text Generation with Shakespeare Dataset

Make model that write like **Shakespeare** character by character.  

It learn from Shakespeare text and do generation 1 char at one time.

---

# Generative ai

This section have **2 projects** to help you with generative AI stuffs with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### 🤖 Agentic RAG Notebook using Llama2 and ChromaDB

This sample make **agentic RAG pipeline** that uses **Llama2** + **ChromaDB**.  

It answer smart questions and check if documents are needed or not, before giving answers. Gives accurate result.

### 📊 Auto Evaluation with Structure Outputs

**Automated evaluation with Structured Output** use a **Meta Llama 2** model to make a MLflow API that score text batch using some rubric.

* Use `llama.cpp` local (no data go outside)
* Add it to MLflow pyfunc
* Has REST endpoint for calling
* Includes two UIs — **Streamlit** and HTML/JS page with csv download.

---

# Nvidia gpu cloud

This parts show how to use **NVIDIA NGC** with [**Z by HP AI Studio**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

You get **4 projects**, fast and simple.

### 🎙️ Audio Translation with NeMo

Do complete **audio translate** using **NVIDIA NeMo** models. Steps:

1. **STT** with Citrinet  
2. **Translate** to Spanish  
3. **TTS** using FastPitch and HiFiGAN  

All runs on GPU. Integrated with **MLflow**.

### 📡 OpenCellID EDA with Panel and cuDF

GPU-powered **explore data dashboard** using [OpenCellID](https://www.opencellid.org/). Made with **Panel** + **cuDF**.

Shows towers by country, operator, etc. on map. Render with GPU power.

### 📈 Stock Analysis with Pandas and cuDF  

This project test time diff between **pandas** (cpu) and **cuDF** (gpu). Two format:

- One notebook from **NVIDIA** that download and eval all-in-one  
- Two notebook style: 1 with Pandas, 1 with cuDF. Metrics log to **MLflow**

### 🌍 Travel Recommend Agent with NeMo and BERT

An AI agent that gives **travel recommends** based on your query.  

It use **BERT** and **NeMo** to guess user intent and reply with smart ideas.

---

# Contact and Support  

- Issue: Make an issue in [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints)

- Docs: See [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview) to fix problem. 

- Community: Come to [**HP AI Creator Community**](https://community.datascience.hp.com/) if you got question.

---

> Build with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
