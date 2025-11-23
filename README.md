# EmoCare: Multimodal Emotion Comfort Companion

**Group Information:**
- **Student ID:** 23097738d
- **Name:** CHEN Hongli

---

## Project Overview

EmoCare is an advanced multimodal AI companion designed to provide empathetic support through natural conversation. By integrating text, audio, and visual analysis, the system detects the user's emotional state in real-time and responds with context-aware, comforting dialogue and expressive speech. The project aims to bridge the gap between technical interaction and human-like empathy, offering a non-clinical support system that "sees," "hears," and "understands" the user while prioritizing privacy by processing sensitive emotional data locally on the device.

---

## Additional Features Implemented

I have implemented a comprehensive multimodal system that combines state-of-the-art deep learning models with an interactive user interface.

### 1. Libraries Used

The project leverages a robust stack of Python libraries for deep learning, audio processing, and interface design:

*   **`transformers`, `peft`, `bitsandbytes`**: For loading and fine-tuning the Large Language Model (Llama 3.1) with efficient 4-bit quantization and LoRA adapters.
*   **`torch`, `torchvision`**: The core deep learning framework used for the visual emotion recognition model (ResNet18) and tensor operations.
*   **`speechbrain`**: Used for the audio emotion recognition system, leveraging pre-trained wav2vec2 models.
*   **`edge-tts`**: Provides the Text-to-Speech functionality with dynamic pitch and rate control for expressive voice output.
*   **`azure-cognitiveservices-speech`**: Implements high-accuracy Speech-to-Text (STT) to transcribe user audio input.
*   **`gradio`**: Powers the web-based user interface, enabling real-time interaction with chat, audio, and video inputs.
*   **`soundfile`, `librosa`**: Utilities for reading, writing, and processing audio files.
*   **`Pillow` (PIL)**: Handles image processing for the facial expression recognition pipeline.

### 2. AI Models Implemented

I have integrated and fine-tuned multiple specialized AI models to handle different modalities:

*   **Empathetic Chat Agent (LLM):**
    *   **Model:** `nreHieW/Llama-3.1-8B-Instruct` (Base) + Custom LoRA Adapter.
    *   **Role:** Generates warm, concise, and validating responses. The model is loaded in 4-bit quantization (NF4) to run on consumer GPUs, and the LoRA adapter steers it towards a non-clinical, supportive persona.

*   **Text Emotion Recognition:**
    *   **Model:** `j-hartmann/emotion-english-distilroberta-base`.
    *   **Role:** Analyzes the sentiment of the user's text input to detect emotions like joy, sadness, anger, and fear.

*   **Audio Emotion Recognition:**
    *   **Model:** `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`.
    *   **Role:** extracts emotional cues directly from the user's voice tone/prosody, identifying states like anger, happiness, or sadness independent of the spoken words.

*   **Visual Emotion Recognition:**
    *   **Model:** ResNet18 (Custom trained on facial expression datasets).
    *   **Role:** Analyzes the user's webcam feed to detect facial expressions, providing a third layer of emotional context.

### 3. Key Features Implemented

*   **Text-to-Speech (TTS) with Tone Variations and Speech-to-Text (STT):**
    *   Unlike standard robotic TTS, this system modulates the voice based on the context.
    *   **Dynamic Adjustment:** If the system detects "sadness," the voice becomes softer and slower. If "joy" is detected, the pitch and rate increase for a cheerful tone.

*   **Multimodal Emotion Fusion:**
    *   A weighted fusion algorithm aggregates probabilities from text, audio, and visual models to determine the single most dominant emotion, ensuring higher accuracy than relying on a single modality.

*   **Real-Time Interaction:**
    *   Users can interact via text, voice (microphone), or video (webcam) simultaneously. The system handles all inputs in a single inference step.

*   **Avatar System with Emotion-Based Changes:**
    *   The user interface features a dynamic companion avatar that visually reacts to the conversation.
    *   The avatar's expression (Neutral, Happy, Sad, Angry, Anxious) updates automatically based on the **fused emotion score**, which combines confidence levels from text, audio, and visual inputs.

### 4. Privacy & Data Security

*   **Local Processing:** All sensitive user data—including webcam video feeds, microphone audio recordings for emotion analysis, and text chat logs—are processed entirely locally on your device. The core AI models (LLM, Facial Emotion, Audio Emotion) run offline to ensure your privacy.
*   **External Services Exception:** The system uses secure external cloud services solely for **Speech-to-Text (Azure)** and **Text-to-Speech (Edge TTS)** to ensure high-quality voice interaction. Audio data sent to these services is used only for momentary processing and is not stored by this application.

---

## System Requirements

*   **Hardware:** 
    *   **GPU:** NVIDIA RTX 4060 with 8GB VRAM (Recommended for 4-bit LLM inference).
    *   **RAM:** 16GB or higher.
    *   **Storage:** Approx 10GB free space for model weights.
*   **OS:** Windows 10/11 (tested) or Linux.
*   **Software:** Python 3.10 or higher.

---

## Installation and Setup

Follow these steps to set up the project environment:

1.  **Clone or Extract the Project:**
    Ensure the project files are in a local directory (e.g., `.\emocare`).

2.  **Configuration Requirements:**
    *   **Environment Variables:**
        *   Create a `.env` file in the root directory if needed (or edit `backend/config.py`).
        *   **Hugging Face Token:** Ensure you have a token with access to Llama 3.1 and set it as `HF_TOKEN`.
        *   **Azure Speech:** Set `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in `backend/config.py` to enable the Speech-to-Text feature.
    *   **Model Downloads:** The system will automatically download required models from Hugging Face on the first run. Ensure you have a stable internet connection.

3.  **Create a Virtual Environment (Optional but Recommended):**
    `python -m venv .venv`
    `.venv\Scripts\activate`

4.  **Install Dependencies:**
    Install all required Python libraries:
    `pip install -r requirements.txt`
        *Note: If you encounter issues with `torch`, ensure you install the CUDA-enabled version compatible with your GPU from [pytorch.org](https://pytorch.org/).*

---

## How to Run

    `python app.py`

### Expected Outputs

1.  **Console Output:** You should see logs indicating the loading of models (SpeechBrain, DistilRoBERTa, Llama 3.1) and a success message:
    Running on local URL:  `http://localhost:9001`
2.  **Web Interface:**
    *   Open your web browser and go to `http://localhost:9001`.
    *   You will see the EmoCare interface with the Chatbot, Avatar, and Input controls.
3.  **Interaction:**
    *   **Type:** "I'm feeling really stressed about my exams." -> The Avatar should look concerned, and the text reply will be supportive.
    *   **Speak:** Use the microphone to say something. The text will be transcribed, and the system will analyze your voice tone.
    *   **Show Face:** Enable your webcam to let the companion see your facial expressions. A smile or frown will be detected and fused with other inputs to refine the emotional response.
    *   **Audio Response:** The companion will reply with voice audio that matches the emotional context.
