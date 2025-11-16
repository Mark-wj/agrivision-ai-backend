# 🌱 AgroVision AI - Backend API

> AI-powered plant disease detection system backend built with FastAPI and TensorFlow

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)


## 🎯 Overview

This is the backend API for AgroVision AI, providing real-time plant disease detection through deep learning. The system analyzes plant leaf images and returns disease diagnoses with treatment recommendations.

**Key Features:**
- 🔬 Detects 38 different plant diseases
- 🚀 Fast inference (~500ms per prediction)
- 📊 94%+ accuracy on validation data
- 💊 Provides treatment recommendations
- 🌍 RESTful API with interactive documentation

**Supporting UN SDG 2: Zero Hunger** 🌾

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI REST API                │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────────┐    │
│  │  Image   │→ │    Prediction    │    │
│  │ Processing│  │     Engine       │    │
│  └──────────┘  └──────────────────┘    │
├─────────────────────────────────────────┤
│     MobileNetV2 + Custom Classifier     │
│         (TensorFlow/Keras)              │
└─────────────────────────────────────────┘
```

**Model:** MobileNetV2 (Transfer Learning)  
**Input:** 224x224 RGB images  
**Output:** 38-class probability distribution

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10-3.12
- 8GB RAM minimum
- 2GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/Mark-wj/agrovision-backend.git
cd agrovision-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start server
python main.py
```

Server runs on **http://localhost:8000**

### API Documentation

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_loaded": true,
  "total_classes": 38
}
```

### Get Disease Classes
```http
GET /classes
```

**Response:**
```json
{
  "total_classes": 38,
  "classes": ["Apple___Apple_scab", "Tomato___healthy", ...],
  "organized_by_crop": {
    "Apple": ["Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"],
    "Tomato": [...]
  }
}
```

### Predict Disease
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "disease": "Tomato___Early_blight",
    "confidence": 0.9567,
    "is_healthy": false,
    "severity": "Moderate to High",
    "description": "Fungal infection detected..."
  },
  "recommendations": [
    "Apply appropriate fungicide treatment immediately",
    "Remove and dispose of infected leaves",
    ...
  ],
  "alternative_predictions": [
    {"disease": "Tomato___Early_blight", "confidence": 0.9567},
    {"disease": "Tomato___Late_blight", "confidence": 0.0312}
  ]
}
```

---

## 🧠 Model Details

### Architecture
- **Base Model:** MobileNetV2 (ImageNet pre-trained)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense (512 units, ReLU)
  - Dropout (0.3)
  - Dense (38 units, Softmax)

### Training
- **Dataset:** PlantVillage (~87,000 images)
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Categorical Crossentropy
- **Epochs:** 20
- **Batch Size:** 32
- **Data Augmentation:** Rotation, shift, zoom, flip

### Performance
| Metric | Value |
|--------|-------|
| Training Accuracy | 96.45% |
| Validation Accuracy | 94.23% |
| Inference Time | ~500ms |
| Model Size | 11.2 MB |
| Parameters | 2,933,350 |

---

## 🌾 Supported Crops & Diseases

### Apple (4)
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Corn (4)
- Cercospora Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape (4)
- Black Rot
- Esca (Black Measles)
- Leaf Blight
- Healthy

### Tomato (10)
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Yellow Leaf Curl Virus
- Healthy

*...and more (38 classes total)*

---

## 🛠️ Development

### Project Structure

```
agrovision-backend/
├── main.py                    # FastAPI application
├── model_builder.py           # Model loading utilities
├── requirements.txt           # Python dependencies
├── agrovision_model_best.h5   # Trained model (Git LFS)
├── class_indices.json         # Disease class mappings
├── .gitignore
├── .gitattributes             # Git LFS configuration
├── render.yaml                # Render deployment config
└── README.md
```

### Running Tests

```bash
# Test model loading
python model_builder.py

```

### Adding New Features

1. Create feature branch
```bash
git checkout -b feature/new-endpoint
```

2. Make changes and test locally
```bash
python main.py
```

3. Commit and push
```bash
git add .
git commit -m "Add new endpoint"
git push origin feature/new-endpoint
```

4. Create pull request on GitHub

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions focused and small

---


## 🙏 Acknowledgments

- **Dataset:** PlantVillage Dataset by Penn State University
- **Framework:** FastAPI by Sebastián Ramírez
- **ML Framework:** TensorFlow by Google
- **Model Architecture:** MobileNetV2

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Mark-wj/agrovision-backend/issues)
- **Email:** marknjenga25@gmail.com.com
- **Documentation:** [API Docs](https://agrivision-ai-backend.onrender.com/docs)

---

## 📈 Stats

![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/agrovision-backend?style=social)
![GitHub Forks](https://img.shields.io/github/forks/YOUR_USERNAME/agrovision-backend?style=social)
![GitHub Issues](https://img.shields.io/github/issues/YOUR_USERNAME/agrovision-backend)

---

**Built with ❤️ for sustainable agriculture**

Supporting **UN SDG 2: Zero Hunger** 🌾
