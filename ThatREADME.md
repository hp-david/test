<h1 style="text-align: center; font-size: 40px;"> Deep Learning Blueprint Projects for HP AI Studio </h1>

In this folder, we move forward with valuable examples Deep Learning applications on AI Studio, focusing mainly in Computer Vision with Convolutional Neural Networks and Natural Language with Transformers. Currently, we are working with three examples, all of them requiring Deep Learning workspaces (preferably with GPU) to run.

## Repository Structure

The repository is organized into the following structure:

```
├── image_super_resolution_with_FSRCNN/               # Image Super Resolution Template
│   ├── notebooks/                                    # Contains the notebook files
│   │   └── image_super_resolution_with_FSRCNN.ipynb  # Main notebook for FSRCNN
│   └── README.md                                     # Detailed documentation for the super resolution example
│
├── Question and Answer with BERT/                    # Question and Answer Template
│   ├── code/                                         # Demo code
│   ├── notebooks/                                    # Contains the notebook files
│   │   ├── Testing Mlflow Server.ipynb               # Notebook for testing the MLflow server
│   │   ├── question_answering_with_BERT.ipynb        # Main notebook for the project
│   │   └── deploy.py                                 # Code to deploy
│   ├── README.md                                     # Detailed documentation for the Q&A example
│   └── requirements.txt                              # Dependencies for the Q&A example
│
├── text_generation_with_RNN/                         # Text Generation Template
│   ├── code/                                         # Demo code
│
│   ├── data/                                         # Data assets used in the project
│   │   └── shakespeare.txt                           # Text from Shakespeare's Sonnet 1 used in this template
│
│   ├── demo/                                         # Compiled interface folder
│
│   ├── notebooks/                                    # Contains the notebook files
│   │   ├── models/                                   # Trained models and components
│   │   │   ├── decoder.pt                            # Reconstructs input data from compressed form
│   │   │   ├── dict_torch_rnn_model.pt               # Trained model for Torch notebook
│   │   │   ├── encoder.pt                            # Compresses input into compact representation
│   │   │   └── tf_rnn_model.h5                       # Trained model for TensorFlow notebook
│   │   ├── Deployment.ipynb                          # Notebook for registering the model using MLflow
│   │   ├── text_generation_with_RNN_TF.ipynb         # Notebook for the TensorFlow trained model
│   │   ├── text_generation_with_RNN_Torch.ipynb      # Notebook for the Torch trained model
│   │   └── deploy.py                                 # Code to deploy
│
│   └── README.md                                     # Project documentation
```
## 1. Bert QA
This experiment shows a simple BertQA experiment, providing code to train a model, and other to load a trained model from Hugging Face, deploying a service in MLFlow to perform the inference

## 2. Text Generation
This experiment shows how to create a simple text generation, one character per time. This example uses a dataset of Shakespeare's texts.

## 3. Super Resolution
This is a Computer Vision experiment that uses convolutional networks for image transformation - more specifically improving the resolution of an image. This experiment requires the DIV2K dataset to run, that should be downloaded from s3://dsp-demo-bucket/div2k-data into an assset called DIV2K.


# Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
