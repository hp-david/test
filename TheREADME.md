# Generative AI

The sample project in this folder demonstrate how too build generative AI app with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provides **7 blueprint project**, each desinged for quick and easy use to helps you gets started efficiently.


### 📈 Data Analysis with cuDF  

In this project, we provides notebooks to comparing the execution times of dataset operation using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelarated drop-in replace for Pandas. This example is present in two different format:

- **Original Example Notebook**: This version, created by NVIDIA, run the entire evaluation within single notebook. It includes download the data and restart the kernel to activate cuDF extension.

- **Data Analysis Notebooks**: These notebook use preprocessed dataset of varying sizes from **datafabric** folder in AI Studio. The evaluation split across two notebook — one using Pandas (CPU) and other using cuDF (GPU) — with performance metrics log to **MLflow**.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
