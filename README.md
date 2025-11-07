# DEEPACE2.0
# 🎧 DeepACE2.0 — Deep Learning–Based Cochlear Implant Signal Processing

## 🧩 Overview
This project implements a **deep learning–based sound coding strategy** for **cochlear implants**, inspired by the **DeepACE2.0 framework**.  
The system replaces part of the conventional **ACE (Advanced Combination Encoder)** pipeline with a **1D Convolutional Neural Network (CNN)** that learns to predict **electrodograms** directly from raw audio waveforms.

The objective is to enhance **speech perception** for cochlear implant users, especially in **noisy acoustic environments**, by learning a data-driven mapping between acoustic and electrical domains.

---

## 🚀 Objectives
- Develop a deep neural model to predict electrodograms from speech signals.
- Improve speech intelligibility and robustness to noise compared to ACE.
- Minimize **Mean Squared Error (MSE)** between predicted and true electrodograms.

---

## 🧠 Key Features
- **Signal Preprocessing**
  - Used **MATLAB + NMT Toolbox** to preprocess raw audio files.
  - Generated `.mat` files containing `x` (audio signal) and `fs` (sampling rate).

- **DeepACE Model**
  - Implemented **1D CNN** for end-to-end mapping of temporal audio data to stimulation sequences.
  - Supports configurable input length, stride, and kernel size.
  - Optimized using **MSE loss** to ensure accurate temporal predictions.

- **Training & Evaluation**
  - Training performed on preprocessed audio-electrodogram pairs.
  - Evaluated with metrics like **MSE**, **correlation**, and **perceptual similarity**.
  - Compared model predictions against traditional ACE outputs.

- **Visualization**
  - Generated **dot-style electrodograms** to visualize electrode activation patterns.
  - Used spectrogram overlays for qualitative analysis.

---

## 🧰 Tools & Technologies
| Category | Tools |
|-----------|--------|
| **Languages** | Python, MATLAB |
| **Frameworks** | PyTorch |
| **Libraries** | NumPy, SciPy, Matplotlib, Librosa, h5py |
| **Toolboxes** | NMT Toolbox (for preprocessing) |
| **Environment** | Jupyter Notebook / Command Line |

---

## 📂 Repository Structure
