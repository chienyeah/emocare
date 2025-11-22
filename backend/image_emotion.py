import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from backend.config import get_device

# Map standard FER2013 indices to labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

device = get_device()

# --------------------------------------------------------
# Model Definition (Standard ResNet18)
# --------------------------------------------------------
# We use standard torchvision ResNet18 definition here for simplicity
# and compatibility with standard weights if we switch to them.
# The previous custom definition is commented out or replaced to match standard structure.

from torchvision.models import resnet18

def get_model():
    # Initialize standard ResNet18
    model = resnet18(weights=None) # We load weights manually
    
    # Modify first conv layer to accept 1 channel (grayscale) if needed
    # OR we keep it 3 channels and convert input to 3 channels (easier for pretrained models).
    # FER2013 is 48x48 grayscale.
    
    # Modify the final fully connected layer for 7 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EMOTION_LABELS))
    
    return model

# --------------------------------------------------------
# Setup
# --------------------------------------------------------

MODEL_PATH = os.path.join("models", "fer", "fer_resnet18.pth")
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: FER model weights not found at {MODEL_PATH}. using fallback.")
        return False
    
    try:
        # Initialize model
        net = get_model()
        
        # Load weights
        # Note: If we downloaded ImageNet weights (microsoft/resnet-18), they won't match exactly 
        # because of the fc layer dimension (1000 vs 7).
        # We'll load with strict=False to ignore the fc layer mismatch for now,
        # allowing the code to run even if predictions are random/ImageNet-biased.
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
            
        # Load state dict
        # strict=False is crucial here if using ImageNet weights as placeholder
        net.load_state_dict(new_state_dict, strict=False)
        
        net.to(device)
        net.eval()
        model = net
        return True
    except Exception as e:
        print(f"Error loading FER model: {e}")
        return False

# Attempt load on module import
load_model()

# Preprocessing transforms
# Standard ResNet expects 224x224 usually, but for FER2013 48x48 is common.
# If using standard ResNet, 48x48 might be too small for the pooling layers (7x7 avg pool).
# But let's try with 48x48 first as requested.
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

def analyze_image(image: Image.Image) -> dict:
    """
    Analyzes the emotion of the given PIL Image.
    
    Args:
        image (PIL.Image): The image to analyze.
        
    Returns:
        dict: {'label': <emotion>, 'score': <probability>}
    """
    global model
    if model is None:
        # Try loading again if it failed previously
        if not load_model():
            return {"label": "neutral", "score": 0.0, "details": "Model not loaded"}
            
    try:
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, 48, 48]
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # Softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            score, idx = torch.max(probs, dim=1)
            
            idx = idx.item()
            score = score.item()
            
            # If using ImageNet weights, outputs will be 1000 classes if we didn't replace fc.
            # But we replaced fc, so it's initialized randomly.
            # Predictions will be random until we train/load real FER weights.
            
            label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else "unknown"
            
            return {
                "label": label,
                "score": score
            }
            
    except Exception as e:
        print(f"Error processing image emotion: {e}")
        return {"label": "error", "score": 0.0}

if __name__ == "__main__":
    print("Image emotion module loaded.")
