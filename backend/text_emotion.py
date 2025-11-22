import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.config import get_device, HF_TOKEN

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
device = get_device()

# Load model and tokenizer once when the module is imported
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(device)
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    raise e

def analyze_text(text: str) -> dict:
    """
    Analyzes the emotion of the given text using a pre-trained DistilRoBERTa model.
    
    Args:
        text (str): The text to analyze.
        
    Returns:
        dict: A dictionary containing the predicted emotion label and its probability score.
              Example: {'label': 'joy', 'score': 0.95}
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Get the highest scoring label and its probability
    score, label_idx = torch.max(probs, dim=-1)
    
    label_idx = label_idx.item()
    score = score.item()
    
    label = model.config.id2label[label_idx]
    
    return {
        "label": label,
        "score": score
    }

if __name__ == "__main__":
    # Simple test
    test_text = "I am feeling wonderful today!"
    print(f"Analyzing: '{test_text}'")
    result = analyze_text(test_text)
    print(f"Result: {result}")

