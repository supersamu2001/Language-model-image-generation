# Computer Vision & Language Model Project - Group Rosenblatt

<p align="center">
  <img src="https://www.ifrax.it/scuola.png" alt="University of Pisa Logo" width="200">
</p>

This project was developed as part of the **Multimedia Information Retrieval and Computer Vision** course (A.Y. 2025-2026, Prof. Nicola Tonellotto) for the Master's Degree in *Artificial Intelligence and Data Engineering* at the University of Pisa.

**Authors (Group Rosenblatt):**
* Nilo Fabiano
* Gabriele Frassi
* Samuele Marchi
* Lorenzo Valtriani

## 🚀 Project Overview
The objective of this project is to develop an advanced **Image Generation System** focused on **spatial completion** (specifically reconstructing the right-hand side of an image given its left-hand context). 

Inspired by generative tools like *Photoshop’s Generative Fill* and *OpenAI’s ImageGPT*, our system leverages a **Two-Stage Architecture** to treat image pixels as discrete tokens, enabling the use of powerful Language Modeling techniques for visual synthesis.

### Key Features
* **Two-Stage Pipeline:** Decoupling visual representation (VQ-VAE) from generative modeling (GPT-based Transformer).
* **Spatial Completion:** Specialized in predicting the "future" (right side) of an image starting from a visual prompt (left side).
* **Latent Space Discretization:** Transformation of continuous images into a structured codebook of discrete latent tokens.

## 🛠 Technical Architecture

### 1. Stage I: VQ-VAE (Vector Quantized Variational Autoencoder)
To bridge the gap between continuous visual data and discrete sequence modeling:
* **Encoder-Decoder:** Compresses images into a downsampled grid of latent vectors.
* **Vector Quantization:** Maps latent vectors to the nearest entries in a learned **Codebook**, effectively "tokenizing" the image.
* **Reconstruction:** Ensures the codebook preserves high-fidelity visual information.

### 2. Stage II: Transformer Language Model (GPT)
The generative core of the system:
* **Autoregressive Prediction:** A Transformer model trained to predict the next visual token in a sequence, following the GPT architecture.
* **Spatial Logic:** The model learns the statistical dependencies between visual tokens, allowing it to complete missing portions of an image coherently.

## 📁 Notebook Structure
The `CV LM Project - Rosenblatt.ipynb` notebook follows a structured workflow:
1.  **Environment Setting:** Setup of PyTorch, Torchvision, and specialized CV libraries.
2.  **VQ-VAE Training:** Implementation of the quantization layer and reconstruction loss.
3.  **Codebook Analysis:** Visualization of the learned discrete representations.
4.  **Transformer Implementation:** Building the GPT-like model for token sequences.
5.  **Inference & Completion:** Generating the right-hand side of test images and evaluating visual coherence.

## ⚙️ Requirements
The project is designed to be executed in a GPU-accelerated environment (NVIDIA Tesla T4 or better recommended):
* `PyTorch` & `torchvision`
* `tqdm` (for progress tracking)
* `matplotlib` & `PIL` (for image processing and visualization)

## 📖 Usage
The notebook is self-contained and includes:
* **Training Modules:** Full code to train both the VQ-VAE and the Transformer from scratch.
* **Visualization Tools:** Functions to display original images vs. reconstructions and the final spatial completions.
* **Pre-trained Options:** (If applicable) Logic to load saved weights for quick testing of the generative capabilities.
