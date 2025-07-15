# Interactive ORPO Fine-Tuning & Inference Hub for Open LLMs

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![HuggingFace](https://img.shields.io/badge/Hugging--Face-model-yellow.svg?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-used-ee4c2c.svg?logo=pytorch)
![ORPO](https://img.shields.io/badge/ORPO-fine--tuning-lightblue.svg)

</div>

## 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview

This project demonstrates a full-stack LLM fine-tuning experiment using ORPO (Open-Source Reinforcement Pretraining Objective) to align a base language model with human preference data. It leverages the **Z by HP AI Studio Local GenAI environment**, and uses models such as LLaMA 3, Gemma 1B, and Mistral 7B as foundations.

We incorporate:

- **TensorBoard** for human feedback visualization before fine-tuning
- A flexible model selector and inference runner architecture
- A comparative setup to benchmark base vs fine-tuned models on the same prompts
- Detailed model comparison tools for quality evaluation

---

## Project Structure

```
├── config                                         # Configuration files
│   ├── default_config_cpu.yaml
│   ├── default_config_multi-gpu.yaml
│   └── default_config_one-gpu.yaml
│
├── core                                           # Core Python modules
│   ├── comparer
│   │   └── model_comparer.py
│   ├── data_visualizer
│   │   └── feedback_visualizer.py
│   ├── deploy
│   │   └── deploy_fine_tuning.py
│   ├── finetuning_inference
│   │   └── inference_runner.py
│   ├── local_inference
│   │   └── inference.py
│   ├── ggml_convert
│   │   └── convert-lora-to-ggml.py
│   │   └── convert.py
│   ├── selection
│   │   └── model_selection.py
│   ├── target_mapper
│   │   └── lora_target_mapper.py
│
├── notebooks
│   └── fine_tuning_orpo.ipynb                   # Main notebook for the project
│
├── README.md                                    # Project documentation
└── requirements.txt                             # Required dependencies

```

---

## Setup

### Step 0: Minimum Hardware Requirements

To ensure smooth execution and reliable model deployment, make sure your system meets the following **minimum hardware specifications** based on the selected model and task (inference or fine-tuning):

### ✅ Model Hardware Matrix

| **Model**                             | **Task**    | **Min VRAM** | **Min RAM** | **GPU Recommendation**           |
| ------------------------------------- | ----------- | ------------ | ----------- | -------------------------------- |
| `mistralai/Mistral-7B-Instruct-v0.1`  | Inference   | 12 GB        | 32 GB       | RTX 3080, A100 (for 4-bit QLoRA) |
|                                       | Fine-tuning | 40–48+ GB    | 64+ GB      | RTX 4090, A100, H100             |
| `meta-llama/Llama-2-7b-chat-hf`       | Inference   | 6 GB         | 32 GB       | RTX 3080 or better               |
|                                       | Fine-tuning | 40–48+ GB    | 64+ GB      | RTX 4090+                        |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Inference   | 16 GB        | 32 GB       | RTX 3090, 4090                   |
|                                       | Fine-tuning | 64+ GB       | 64–96 GB    | Dual RTX 4090 or A100            |
| `google/gemma-7b-it`                  | Inference   | 12 GB        | 32 GB       | RTX 3080 or better               |
|                                       | Fine-tuning | 40+ GB       | 64 GB       | RTX 4090                         |
| `google/gemma-3-1b-it`                | Inference   | 8 GB         | 16–24 GB    | RTX 3060 or better               |
|                                       | Fine-tuning | 16–24 GB     | 32–48 GB    | RTX 3080 / 3090                  |

> ⚠️ These recommendations are based on community benchmarks and documentation provided by Hugging Face, Unsloth, and Google. For production workloads, always monitor VRAM/RAM usage on your system.

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

```bash
git clone https://github.com/HPInc/AI-Blueprints.git

```

- Ensure all files are available after workspace creation.

### Step 4: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file located in the `configs` folder:
  - `HUGGINGFACE_API_KEY`: Required to use Hugging Face-hosted models instead of a local LLaMA model.
- Edit `config.yaml` with relevant configuration details.

---

## Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/fine_tuning_orpo.ipynb
```

This will:

- Select and download a compatible model from Hugging Face
- Apply QLoRA configuration and prepare the model for training
- Run the fine-tuning using ORPO
- Perform evaluation and comparison between the base and fine-tuned models
- Register and serve both base and fine-tuned models via MLflow

### Step 2: Deploy the Chatbot Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.

---

## Contact and Support

- Issues & Bugs: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: [**AI Studio Documentation**](https://zdocs.datascience.hp.com/docs/aistudio/overview).

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**Z by HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
