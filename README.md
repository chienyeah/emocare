# EmoCare

**Student ID:** 23097738d  
**Name:** CHEN Hongli

## Project Overview
EmoCare is a multimodal emotion comfort companion designed to provide empathetic support through natural conversation. It integrates text, audio, and visual cues to detect the user's emotional state and responds with comforting, context-aware dialogue and expressive speech.

## Additional Features & Implementation

I have implemented a comprehensive multimodal system that goes beyond simple text analysis. The key features include:

### 1. Multimodal Emotion Recognition
The system fuses data from three modalities to determine the user's emotional state:
*   **Text Emotion:** Analyzes user messages using a fine-tuned **DistilRoBERTa** model (`j-hartmann/emotion-english-distilroberta-base`) to detect emotions like joy, sadness, anger, etc.
*   **Audio Emotion:** Processes speech input using a **SpeechBrain** model based on **Wav2Vec2** (`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`), capable of recognizing emotions directly from voice tone (neutral, anger, happiness, sadness).
*   **Visual Emotion:** Uses a **ResNet18** CNN trained on facial expression datasets (FER2013/CK+) to analyze facial expressions from the webcam feed.
*   **Fusion Logic:** A weighted fusion algorithm combines confidence scores from all active modalities to determine the dominant emotion.

### 2. Empathetic Comfort Agent (LLM with LoRA)
Instead of generic responses, I generate empathetic, non-clinical support using a Large Language Model:
*   **Base Model:** **Llama 3.1 8B Instruct**, loaded with 4-bit quantization (NF4) via `bitsandbytes` for efficient inference on consumer GPUs.
*   **Fine-tuning:** I integrated a **LoRA (Low-Rank Adaptation)** adapter (trained on comfort-specific datasets) using the `peft` library to steer the model towards warm, concise, and supportive responses.

### 3. Expressive Voice Interaction
*   **Speech-to-Text (STT):** Integrated **Azure Cognitive Services Speech SDK** for accurate, real-time transcription of user speech.
*   **Text-to-Speech (TTS):** Implemented **Edge TTS** (`edge-tts`) with dynamic voice modulation. The system changes the voice pitch, rate, and style (e.g., cheerful, empathetic, serious) to match the generated response's emotional context.

### 4. Interactive Avatar & UI
*   **Gradio Interface:** A user-friendly web UI allows users to type, speak, or show their face.
*   **Dynamic Avatar:** The companion avatar changes expressions (neutral, happy, sad, angry, anxious) dynamically based on the fused emotion detected during the conversation.

---

## Libraries and Models Used

*   **Deep Learning & NLP:** `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`.
*   **Audio Processing:** `speechbrain`, `torchaudio`, `librosa`, `soundfile`.
*   **Speech Services:** `azure-cognitiveservices-speech` (STT), `edge-tts` (TTS).
*   **Computer Vision:** `torchvision`, `opencv-python`, `Pillow`.
*   **Interface:** `gradio`.

---

## System Requirements

*   **GPU:** NVIDIA RTX 4060 with 8GB VRAM (or better) is **highly recommended** to run the 4-bit quantized Llama 3.1 model and emotion classifiers smoothly.
*   **OS:** Windows 10/11 or Linux.
*   **Python:** 3.10+.

## How to Run the Source Code

1.  **Install Dependencies:**
    Ensure you have Python installed, then run:
    pip install -r requirements.txt
    2.  **Configuration:**
    *   Ensure you have a Hugging Face token with access to Llama 3.1 (if required by the base model) to be added in `.env`.
    *   (Optional) Set up your Azure Speech Key and Region in `backend/config.py` or `.env` file for Speech-to-Text functionality.

3.  **Launch the Application:**
    Run the main application script:
    python app.py
    4.  **Access the UI:**
    Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:9001` or port 9002).
