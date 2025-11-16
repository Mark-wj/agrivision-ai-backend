# 🌱 AgroVision AI - Backend API

> AI-powered plant disease detection system backend built with FastAPI and TensorFlow

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
git clone https://github.com/YOUR_USERNAME/agrovision-backend.git
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

# Test API endpoints
python -m pytest tests/  # (if tests are added)
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

## 🚀 Deployment

### Deploy to Render

1. **Push to GitHub** (with Git LFS for model)

2. **Create Web Service** on Render:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Access at:** `https://your-service.onrender.com`

### Environment Variables

No environment variables required for basic setup.

For production, consider:
- `MODEL_PATH` - Custom model file path
- `MAX_UPLOAD_SIZE` - Maximum image upload size
- `LOG_LEVEL` - Logging verbosity

---

## 🔒 Security

- ✅ Input validation on all endpoints
- ✅ File type checking (images only)
- ✅ CORS configured for specific origins
- ✅ HTTPS enforced in production
- ⚠️ Consider adding rate limiting for production

### Recommended: Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_disease(...):
    ...
```

---

## 🐛 Troubleshooting

### Model Not Loading

**Error:** `Layer "dense" expects 1 input(s)...`

**Solution:** Using `model_builder.py` which rebuilds architecture and loads weights separately.

### Out of Memory

**Error:** `ResourceExhaustedError`

**Solution:** Reduce batch size or upgrade server RAM.

### Slow Predictions

**Issue:** First request takes 30+ seconds

**Cause:** Render free tier spins down after inactivity

**Solution:** Upgrade to paid tier or use UptimeRobot to keep alive

---

## 📊 Performance Tips

1. **Image Preprocessing:** Resize images client-side before upload
2. **Caching:** Cache common predictions with Redis
3. **Batch Processing:** Process multiple images in one request
4. **Model Optimization:** Use TensorFlow Lite for faster inference

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

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Dataset:** PlantVillage Dataset by Penn State University
- **Framework:** FastAPI by Sebastián Ramírez
- **ML Framework:** TensorFlow by Google
- **Model Architecture:** MobileNetV2

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/agrovision-backend/issues)
- **Email:** your.email@agrovision.com
- **Documentation:** [API Docs](https://your-api.onrender.com/docs)

---

## 🗺️ Roadmap

- [ ] Add batch prediction endpoint
- [ ] Implement caching layer (Redis)
- [ ] Add rate limiting
- [ ] Support more crop types
- [ ] Multi-language recommendations
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Prometheus metrics export

---

## 📈 Stats

![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/agrovision-backend?style=social)
![GitHub Forks](https://img.shields.io/github/forks/YOUR_USERNAME/agrovision-backend?style=social)
![GitHub Issues](https://img.shields.io/github/issues/YOUR_USERNAME/agrovision-backend)

---

**Built with ❤️ for sustainable agriculture**

Supporting **UN SDG 2: Zero Hunger** 🌾
