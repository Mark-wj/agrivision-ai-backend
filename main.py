"""
AgroVision AI - Backend API
FastAPI server for plant disease detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import uvicorn
import os

# Import our custom model builder to fix TensorFlow version issues
from model_builder import load_model_with_weights

# Initialize FastAPI app
app = FastAPI(
    title="AgroVision AI API",
    description="Plant Disease Detection API using Deep Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  
        "https://agrivision-ai-frontend.vercel.app",
        "https://agrivision-ai-frontendd.vercel.app",  
        "https://*.vercel.app",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and classes
model = None
class_names = {}
disease_recommendations = {}

# Model configuration
MODEL_PATH = 'agrovision_model_best.h5'
CLASS_INDICES_PATH = 'class_indices.json'
IMG_SIZE = 224

def load_disease_recommendations():
    """
    Load disease information and treatment recommendations
    This can be expanded based on agricultural research
    """
    return {
        'healthy': {
            'severity': 'None',
            'description': 'The plant appears healthy with no visible signs of disease.',
            'recommendations': [
                'Continue regular watering and fertilization schedule',
                'Monitor plants regularly for any changes',
                'Maintain good air circulation around plants',
                'Practice crop rotation to prevent future diseases',
                'Remove any dead or yellowing leaves promptly'
            ]
        },
        'bacterial': {
            'severity': 'High',
            'description': 'Bacterial infection detected. Requires immediate attention to prevent spread.',
            'recommendations': [
                'Remove and destroy infected plant parts immediately',
                'Apply copper-based bactericides as directed',
                'Sterilize pruning tools with 70% alcohol between cuts',
                'Improve drainage to reduce moisture around plants',
                'Avoid working with plants when they are wet',
                'Increase spacing between plants for better air circulation',
                'Consider removing severely infected plants to protect others'
            ]
        },
        'fungal': {
            'severity': 'Moderate to High',
            'description': 'Fungal infection detected on the plant. Early treatment is crucial.',
            'recommendations': [
                'Apply appropriate fungicide treatment immediately',
                'Remove and dispose of infected leaves and debris',
                'Increase air circulation around plants',
                'Water at the base of plants, avoid overhead watering',
                'Apply organic mulch to prevent soil splash onto leaves',
                'Ensure proper plant spacing',
                'Apply fungicide according to label instructions',
                'Consider using biological fungicides for organic farming'
            ]
        },
        'viral': {
            'severity': 'High',
            'description': 'Viral infection detected. No cure available, focus on prevention.',
            'recommendations': [
                'Remove and destroy infected plants immediately',
                'Control insect vectors (aphids, whiteflies, thrips)',
                'Use virus-resistant varieties in future plantings',
                'Disinfect all tools after each use',
                'Do not replant in the same location',
                'Monitor surrounding plants closely for symptoms',
                'Use reflective mulches to deter insect vectors',
                'Practice strict sanitation in the growing area'
            ]
        },
        'pest': {
            'severity': 'Moderate',
            'description': 'Pest damage detected on the plant.',
            'recommendations': [
                'Identify the specific pest causing damage',
                'Use appropriate insecticide or organic pest control',
                'Encourage beneficial insects (ladybugs, lacewings)',
                'Remove heavily infested plant parts',
                'Apply neem oil or insecticidal soap for organic control',
                'Use row covers to protect plants',
                'Practice crop rotation to break pest cycles',
                'Maintain plant health to resist pest damage'
            ]
        },
        'default': {
            'severity': 'Moderate to High',
            'description': 'Disease detected. The plant shows symptoms that require attention.',
            'recommendations': [
                'Remove and destroy infected leaves immediately',
                'Improve air circulation around plants',
                'Avoid overhead watering to reduce moisture on leaves',
                'Apply appropriate treatment as recommended for the specific disease',
                'Isolate infected plants to prevent spread',
                'Consult with local agricultural extension service',
                'Consider resistant varieties for future planting',
                'Maintain proper plant spacing and nutrition'
            ]
        }
    }

def load_model_and_classes():
    """Load the trained model and class indices"""
    global model, class_names, disease_recommendations
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
            print("💡 Please ensure agrovision_model_best.h5 is in the backend directory")
            return
        
        # Check if class indices file exists
        if not os.path.exists(CLASS_INDICES_PATH):
            print(f"❌ Error: Class indices file not found at {CLASS_INDICES_PATH}")
            print("💡 Please ensure class_indices.json is in the backend directory")
            return
        
        # Load model using the custom model builder (fixes TF version issues)
        print("📦 Loading model...")
        model = load_model_with_weights(MODEL_PATH, CLASS_INDICES_PATH)
        
        if model is None:
            print("❌ Failed to load model!")
            return
        
        print("✅ Model loaded successfully!")
        
        # Load class indices
        print("📋 Loading class indices...")
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
            # Reverse the dictionary to map index to class name
            class_names = {v: k for k, v in class_indices.items()}
        
        print(f"✅ Loaded {len(class_names)} plant disease classes")
        
        # Load disease recommendations
        disease_recommendations = load_disease_recommendations()
        print("✅ Disease recommendations loaded")
        
        # Print some class names as verification
        print("\n🌿 Sample disease classes:")
        for i, name in enumerate(list(class_names.values())[:5]):
            print(f"   {i+1}. {name}")
        print(f"   ... and {len(class_names)-5} more classes\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

def get_disease_info(class_name: str):
    """Get disease information and recommendations based on class name"""
    class_lower = class_name.lower()
    
    # Check for healthy plants
    if 'healthy' in class_lower:
        return disease_recommendations['healthy']
    
    # Check for bacterial diseases
    elif 'bacterial' in class_lower or 'bacteria' in class_lower:
        return disease_recommendations['bacterial']
    
    # Check for fungal diseases
    elif any(keyword in class_lower for keyword in ['fungal', 'fungus', 'mold', 'mildew', 'rust', 'rot', 'blight', 'scab', 'spot']):
        return disease_recommendations['fungal']
    
    # Check for viral diseases
    elif 'virus' in class_lower or 'viral' in class_lower or 'mosaic' in class_lower or 'curl' in class_lower:
        return disease_recommendations['viral']
    
    # Check for pest damage
    elif 'mite' in class_lower or 'spider' in class_lower or 'insect' in class_lower:
        return disease_recommendations['pest']
    
    # Default recommendations
    else:
        return disease_recommendations['default']

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "="*70)
    print("🌱 AgroVision AI - Backend API Starting")
    print("="*70 + "\n")
    
    load_model_and_classes()
    
    if model is not None:
        print("="*70)
        print("✅ Server ready to accept requests!")
        print("="*70 + "\n")
    else:
        print("="*70)
        print("⚠️ Server started but model not loaded!")
        print("Please check the model files and restart.")
        print("="*70 + "\n")

@app.get("/")
async def root():
    """Root endpoint - Welcome message"""
    return {
        "message": "Welcome to AgroVision AI API",
        "version": "1.0.0",
        "description": "Plant Disease Detection using Deep Learning",
        "endpoints": {
            "/": "GET - This welcome message",
            "/health": "GET - Check API health status",
            "/classes": "GET - Get all supported disease classes",
            "/predict": "POST - Upload image for disease detection",
            "/docs": "GET - Interactive API documentation (Swagger UI)",
            "/redoc": "GET - Alternative API documentation (ReDoc)"
        },
        "usage": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) > 0,
        "total_classes": len(class_names),
        "version": "1.0.0",
        "message": "API is operational" if model is not None else "Model not loaded"
    }

@app.get("/classes")
async def get_classes():
    """Get all disease classes supported by the model"""
    if not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded. Classes unavailable.")
    
    # Organize classes by crop type
    organized_classes = {}
    for idx, class_name in class_names.items():
        # Split by '___' to get crop and disease
        parts = class_name.split('___')
        if len(parts) == 2:
            crop, disease = parts
            if crop not in organized_classes:
                organized_classes[crop] = []
            organized_classes[crop].append(disease)
        else:
            if 'Other' not in organized_classes:
                organized_classes['Other'] = []
            organized_classes['Other'].append(class_name)
    
    return {
        "total_classes": len(class_names),
        "classes": list(class_names.values()),
        "organized_by_crop": organized_classes
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Image file (JPG, JPEG, PNG)
    
    Returns:
        JSON response with disease prediction and recommendations
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please contact administrator."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get original image size
        original_size = image.size
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "disease": class_names[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        # Determine health status
        is_healthy = 'healthy' in predicted_class.lower()
        
        # Build response
        response = {
            "success": True,
            "prediction": {
                "disease": predicted_class,
                "confidence": confidence,
                "is_healthy": is_healthy,
                "severity": disease_info['severity'],
                "description": disease_info['description']
            },
            "recommendations": disease_info['recommendations'],
            "alternative_predictions": top_3_predictions,
            "metadata": {
                "original_image_size": list(original_size),
                "processed_image_size": [IMG_SIZE, IMG_SIZE],
                "model_version": "1.0.0",
                "total_classes": len(class_names)
            }
        }
        
        return JSONResponse(content=response)
        
    except Image.UnidentifiedImageError:
        raise HTTPException(
            status_code=400, 
            detail="Could not identify image file. Please upload a valid image."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/about")
async def about():
    """Information about AgroVision AI"""
    return {
        "name": "AgroVision AI",
        "description": "AI-powered plant disease detection system",
        "version": "1.0.0",
        "supported_crops": [
            "Apple", "Corn", "Grape", "Peach", 
            "Pepper", "Potato", "Strawberry", "Tomato"
        ],
        "model_info": {
            "architecture": "MobileNetV2 (Transfer Learning)",
            "framework": "TensorFlow/Keras",
            "accuracy": "94%+",
            "total_classes": len(class_names) if class_names else 0
        },
        "sdg_alignment": "UN SDG 2: Zero Hunger",
        "contact": "your.email@agrovision-ai.com"
    }

if __name__ == "__main__":
    # Run the server
    print("\n" + "="*70)
    print("🚀 Starting AgroVision AI Backend Server")
    print("="*70)
    print("\n📍 Server will be available at:")
    print("   • http://localhost:8000")
    print("   • http://127.0.0.1:8000")
    print("\n📚 API Documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("\n⏹️  Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )