# Text Summarization with LangChain

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![LangChain](https://img.shields.io/badge/LangChain-used-lightgreen.svg?logo=langchain)
![Streamlit UI](https://img.shields.io/badge/User%20Interface-Streamlit-ff4b4b.svg?logo=streamlit)
![React UI](https://img.shields.io/badge/User%20Interface-React-61DAFB.svg?logo=react)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Configuration](#configuration)
- [🔧 Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

This project demonstrates how to build a semantic chunking and summarization pipeline for texts using **LangChain** and **Sentence Transformers**. It leverages the **Z by HP AI Studio Local GenAI image** and the Meta Llama 3.1 model with 8B parameters to generate concise and contextually accurate summaries from text data.

---

## Project Structure

```text
├── configs
│   └── config.yaml                                                     # Blueprint configuration (UI mode, ports, service settings)
├── core
│   ├── __init__.py
│   └── text_summarization_service.py                                   # Text summarization service implementation
├── data
│   ├── inputs/                                                         # Input data directory
│   └── outputs/                                                        # Generated summaries directory
├── demo
│   ├── static/                                                         # Static HTML UI files
│   └── streamlit/                                                      # Streamlit webapp files
├── docs
│   ├── sample-html-ss.png                                             # HTML UI screenshot
│   ├── sample-html-ui.pdf                                             # HTML UI page
│   ├── sample-streamlit-ss.png                                        # Streamlit UI screenshot
│   └── sample-streamlit-ui.pdf                                        # Streamlit UI page
├── notebooks
│   ├── register-model.ipynb                                           # Model registration notebook
│   └── text-summarization-with-langchain.ipynb                        # Main text summarization notebook
├── src
│   ├── __init__.py
│   └── utils.py                                                        # Utility functions for config loading
├── README.md
└── requirements.txt
```

---

## Configuration

The blueprint uses a centralized configuration system through `configs/config.yaml`:

```yaml
ui:
  mode: streamlit # UI mode: streamlit or static
  ports:
    external: 8501 # External port for UI access
    internal: 8501 # Internal container port
  service:
    timeout: 30 # Service timeout in seconds
    health_check_interval: 5 # Health check interval in seconds
    max_retries: 3 # Maximum retry attempts
```

---

## Setup

### Step 0: Minimum Hardware Requirements

To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- RAM: 32 GB
- VRAM: 6 GB
- GPU: NVIDIA GPU

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).
- (Optional) Add a description and relevant tags.

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

1. Clone the GitHub repository:

   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```

2. Ensure all files are available after workspace creation..

### Step 4: Add the Model to Workspace

1. Download the Meta Llama 3.1 model with 8B parameters via Models tab:

- **Model Name**: `meta-llama3.1-8b-Q8`
- **Model Source**: `AWS S3`
- **S3 URI**: `s3://149536453923-hpaistudio-public-assets/Meta-Llama-3.1-8B-Instruct-Q8_0`
- **Bucket Region**: `us-west-2`
- Make sure that the model is in the `datafabric` folder inside your workspace.

### Step 5: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file under the `configs` folder:
  - `HUGGINGFACE_API_KEY`
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/text-summarization-with-langchain.ipynb
```

This will:

- Set up the semantic chunking pipeline
- Create the summarization chain with LangChain
- Register the model in MLflow

### Step 2: Deploy the Summarization Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.
- Use the API endpoints to generate summaries from your text data.

### Successful Demonstration of the User Interface

![text Summarization Demo UI](docs/ui_summarization.png)

:warning: Current implementation of deployed model **do not** perform the chunking steps: summarization is run directly by the LLM model. In the case of suggested local model (i.e. Llama3.1-8b), texts with more than 1000 words may cause instabilities when summarization is triggered on the UI. We recommend using different models or smaller texts to avoid these problems.

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
