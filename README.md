# 🎨 AI-Powered Image Colorization

> Bringing black & white images to life using Deep Learning & Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-U--Net%20CNN-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

An AI-powered image colorization system that automatically adds realistic colors to grayscale images using a **CNN-based U-Net architecture** and **LAB color space**. Built as a Digital Image Processing project at FAST-NUCES Lahore.

---

## ✨ Features

- 🧠 **Deep Learning Model** — Custom U-Net CNN with 4.3M parameters
- 🎨 **LAB Color Space** — Separates luminance from chrominance for better accuracy
- ⚡ **Fast Inference** — 2-3 sec on CPU, <1 sec on GPU
- 🖥️ **User-Friendly GUI** — Built with Tkinter for easy image upload & saving
- 📉 **81% Loss Reduction** — From 0.8 to 0.15 over 200 epochs

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.0+ |
| Image Processing | scikit-image, Pillow |
| GUI | Tkinter |
| Numerical Computing | NumPy |

---

## 🏗️ Architecture

Encoder:  1 → 64 → 128 → 256 → 512 (Bottleneck)
Decoder:  512 → 256 → 128 → 64 → 2 (ab channels)
Skip Connections: Preserve spatial details
Output: Tanh activation → LAB → RGB

---

## 📊 Results

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Total Loss | 0.8 | 0.15 | **81%** |
| MSE Loss | 0.5 | 0.08 | **84%** |
| L1 Loss | 0.6 | 0.12 | **80%** |
| Color Magnitude | 0.05 | 0.35 | **600%** |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the GUI
```bash
python colorization_gui.py
```

### Train the Model
```bash
python train_lab.py
```

### Test the Model
```bash
python test_lab.py
```

---

## 📁 Project Structure

dipProj/
├── model_lab.py          # U-Net CNN architecture
├── train_lab.py          # Training pipeline
├── test_lab.py           # Testing & evaluation
├── dataset_loader_lab.py # Data preprocessing
├── colorization_gui.py   # Tkinter GUI
├── debug_model.py        # Debugging utilities
├── requirements.txt      # Dependencies
├── images/               # Sample images
└── outputs/              # Colorized results

---

## 🎯 Applications

- 🏛️ Historic image & photo restoration
- 🎬 Classic film colorization
- 🏥 Medical imaging enhancement
- 🎨 Artistic & creative workflows
- 🔍 Forensic imaging

---

## 👩‍💻 Authors

- **Fatima Tahir** — BCS-5A, FAST-NUCES Lahore
- **Amna Nadeem** — BCS-5B, FAST-NUCES Lahore

---

> *Digital Image Processing Project — FAST-NUCES Lahore*