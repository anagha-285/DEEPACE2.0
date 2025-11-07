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

## 📊 Results

### 🧠 Quantitative Metrics
| Metric | Value |
|--------|--------|
| Mean Squared Error (MSE) | **0.0018** |
| Correlation (r) | **0.92** |
| Training Epochs | 30 |
| Learning Rate | 1e-4 |

### 🔍 Observations
- The **DeepACE CNN** effectively learned temporal mapping between speech audio and electrode stimulation patterns.  
- The model showed **fast convergence** with steadily decreasing MSE across epochs.  
- **Predicted electrodograms** closely matched the target patterns, preserving temporal envelope features critical for speech intelligibility.  

### 🎧 Visual Results

#### 1️⃣ Training vs Validation Loss Curve
<img src="results/plots/mse_curve.png" width="600"/>

#### 2️⃣ Predicted vs Target Electrodogram
<img src="results/plots/pred_vs_target_electrodogram.png" width="600"/>

### 🔬 Interpretation
- Predicted electrodograms exhibit clear temporal synchronization with ACE references.  
- Noise robustness improved due to convolutional feature learning across multiple receptive fields.  
- The DeepACE2.0 approach demonstrates strong potential for **enhanced speech coding** in noisy listening environments.

---

## 🧩 Future Work
- Integrate **Transformer layers** for improved contextual modeling.  
- Explore **denoising autoencoders** for more robust stimulation patterns.  
- Conduct **subjective perceptual testing** to validate intelligibility improvements.


---

## 📂 Repository Structure
