import os
import azure.cognitiveservices.speech as speechsdk
from backend.config import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

# Simple cache directory for transcripts if needed, though we mostly process on the fly
CACHE_DIR = os.path.join("assets", "transcribe_audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def transcribe_audio_file(audio_path: str, language: str = "en-US") -> str:
    """
    Transcribes an audio file using Azure Speech Services (one-shot recognition).
    
    Args:
        audio_path (str): Path to the WAV file.
        language (str): Language code (default 'en-US').
        
    Returns:
        str: The recognized text, or an empty string if failed/no speech detected.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print("Azure Speech credentials not found. STT skipped.")
        return ""

    try:
        # Configure speech service
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_recognition_language = language
        
        # Configure audio input from file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        
        # Create recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # Run recognition (single utterance)
        # recognize_once() returns when the first utterance has been recognized
        # For longer audio, we might need continuous recognition, but for short clips this is best.
        
        # Add a timeout to prevent hanging if the service is unresponsive or audio is silent
        # Timeout in seconds (e.g., 10 seconds)
        try:
            # Note: get() takes timeout in seconds? The SDK docs say get() blocks.
            # Python's future.get(timeout). SDK might return a custom Future.
            # Actually, Azure SDK Future.get() does not always accept timeout depending on version.
            # But we can try. If not supported, we rely on service timeout (default is usually long).
            # Let's use a safer approach: standard call, but wrap in try/except if possible.
            # The standard way in Azure SDK for Python is just .get(). 
            # However, to avoid 'Too little data' protocol errors on client disconnect,
            # we want to ensure this returns reasonably fast.
            # Default Azure timeout is ~20s for silence.
            result = speech_recognizer.recognize_once_async().get()
        except Exception as e:
             print(f"Azure STT Async call failed or timed out: {e}")
             return ""
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print(f"No speech could be recognized: {result.no_match_details}")
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
            return ""
            
    except Exception as e:
        print(f"Error in Azure STT: {e}")
        return ""

    return ""
