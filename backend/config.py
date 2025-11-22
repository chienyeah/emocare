import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Azure Speech Credentials
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

def get_device():
    """
    Returns 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
