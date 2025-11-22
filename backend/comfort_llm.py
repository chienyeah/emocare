import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from backend.config import get_device, HF_TOKEN
import os

# Model name - using Llama 3.1 8B Instruct as base, user can swap later
# Ensure you have access to this model on Hugging Face
MODEL_NAME = "nreHieW/Llama-3.1-8B-Instruct"

def get_llm_device_map():
    """
    Determine device map for LLM loading.
    If CUDA is available, use "auto" or specific GPU distribution.
    If CPU only, we might not be able to use 4-bit quantization effectively with bitsandbytes 
    without some hacks, but we'll try "auto" or "cpu".
    """
    if torch.cuda.is_available():
        return "auto"
    return "cpu"

# Configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading comfort LLM: {MODEL_NAME}...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        
        # If on CPU, 4-bit loading might fail or be very slow. 
        # bitsandbytes is primarily for CUDA.
        if torch.cuda.is_available():
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                token=HF_TOKEN
            )
        else:
            print("Warning: CUDA not available. Loading model in full precision (slow/heavy) or skipping.")
            # Fallback for CPU users (might OOM easily with 8B model)
            # Using float32 or smaller model would be better here.
            # For safety, let's try to load without quantization if CPU, but warn.
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="cpu",
                token=HF_TOKEN,
                torch_dtype=torch.float32
            )
            
        _model.eval()
        print("Comfort LLM loaded.")
    except Exception as e:
        print(f"Error loading Comfort LLM: {e}")
        _model = None

SYSTEM_PROMPT = (
    "You are EmoCare, a warm, non-clinical companion. "
    "Validate feelings, be concise, and suggest at most one small, gentle action. "
    "Avoid medical or diagnostic language."
)

def generate_comfort(user_text: str, emotion: str, history: list = None) -> str:
    """
    Generates a comforting response based on the user's input text and detected emotion.
    
    Args:
        user_text (str): The text input from the user.
        emotion (str): The detected emotion label (e.g., 'sad', 'happy').
        history (list): Previous conversation history [(user_msg, bot_msg), ...].
        
    Returns:
        str: A comforting response (approx 40-80 words).
    """
    if _model is None:
        load_model()
        if _model is None:
            return "I'm having trouble thinking right now, but please know I'm here for you."

    if not user_text:
        user_text = "I don't know what to say, I just feel off today."
        
    # Construct prompt using Llama 3 chat template structure if available.
    # We'll use the tokenizer's chat template feature for best compatibility.
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # Add history to context if provided
    if history:
        for user_msg, bot_msg in history:
            # Skip empty history items if any
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    # Add current user message with emotion context
    # We inject the detected emotion into the user message so the model knows context
    messages.append({"role": "user", "content": f"Detected emotion: {emotion}.\nUser: {user_text}\nReply in 40-80 words."})
    
    try:
        input_ids = _tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(_model.device)
        
        terminators = [
            _tokenizer.eos_token_id,
            _tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Create attention mask explicitly
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=120,
                eos_token_id=terminators,
                pad_token_id=_tokenizer.eos_token_id, # Set pad_token_id explicitly
                temperature=0.6,
                top_p=0.9,
                do_sample=True
            )
            
        response = outputs[0][input_ids.shape[-1]:]
        full_response = _tokenizer.decode(response, skip_special_tokens=True)
        
        return full_response.strip()
        
    except Exception as e:
        print(f"Error generating comfort: {e}")
        return "I hear you, and I want to support you. Take a deep breath."

if __name__ == "__main__":
    # Simple test (only if model is loadable/downloaded)
    pass

# Auto-load model on module import if not in __main__ guard context (e.g. when imported by app)
# This ensures the model starts loading immediately when the server starts
if __name__ != "__main__":
    try:
        load_model()
    except:
        pass

