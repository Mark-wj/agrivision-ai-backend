import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import json
import numpy as np
import h5py

def build_model_from_scratch(num_classes=38):
    """Rebuild the exact model architecture that was used in training"""
    
    print(f"🏗️ Building model architecture for {num_classes} classes...")
    
    # Load MobileNetV2 base 
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model 
    base_model.trainable = False
    
    # Build model with exact same architecture as training script
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_model_with_weights(model_path='agrovision_model_best.h5', class_indices_path='class_indices.json'):
    """
    Load model by rebuilding architecture and loading weights only
    This avoids TensorFlow version compatibility issues
    """
    
    try:
        # Get number of classes from class_indices.json
        print("📋 Reading class indices...")
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
            num_classes = len(class_indices)
        
        print(f"✅ Found {num_classes} disease classes")
        
        # Build model architecture
        model = build_model_from_scratch(num_classes)
        
        # Compile model (required before loading weights)
        print("⚙️ Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("🔨 Initializing model layers...")
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        # Now load weights from the .h5 file
        print(f"📥 Loading weights from {model_path}...")
        model.load_weights(model_path)
        
        print("✅ Model weights loaded successfully!")
        print(f"📊 Model has {model.count_params():,} parameters")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test the model builder
    print("🧪 Testing model builder...")
    model = load_model_with_weights()
    if model:
        print("\n✅ Model builder test successful!")
        print("\nModel Summary:")
        model.summary()
    else:
        print("\n❌ Model builder test failed!")
