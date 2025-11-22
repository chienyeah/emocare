import os
import asyncio
import edge_tts
import tempfile
import hashlib

CACHE_DIR = os.path.join("assets", "voice_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default voice settings (female)
DEFAULT_VOICE = "en-US-AriaNeural"
DEFAULT_RATE = "+15%"
DEFAULT_PITCH = "+0Hz"

# Female voice mapping by emotion/mood
EMOTION_VOICE_MAP = {
    "happy": {"voice": "en-US-AriaNeural", "rate": "+20%", "pitch": "+4Hz"},
    "joy": {"voice": "en-US-AriaNeural", "rate": "+20%", "pitch": "+4Hz"},
    "surprised": {"voice": "en-US-AnaNeural", "rate": "+18%", "pitch": "+3Hz"},
    "angry": {"voice": "en-US-JennyNeural", "rate": "+5%", "pitch": "-4Hz"},
    "sad": {"voice": "en-US-JennyNeural", "rate": "-15%", "pitch": "-6Hz"},
    "fear": {"voice": "en-US-AriaNeural", "rate": "-5%", "pitch": "-2Hz"},
    "disgust": {"voice": "en-US-JennyNeural", "rate": "-5%", "pitch": "-2Hz"},
    "neutral": {"voice": "en-US-AriaNeural", "rate": DEFAULT_RATE, "pitch": DEFAULT_PITCH},
}

async def _synthesize_async(text: str, output_path: str, voice: str, rate: str, pitch: str) -> str:
    """
    Async helper to run edge-tts synthesis.
    """
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)
    return output_path

def get_voice_settings(emotion: str):
    """
    Returns voice, rate, pitch settings based on emotion/mood.
    """
    emotion = (emotion or "neutral").lower()
    settings = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])
    return (settings["voice"], settings["rate"], settings["pitch"])


def synthesize_speech(text: str, emotion: str = "neutral") -> str:
    """
    Synthesizes text to speech using edge-tts and returns the path to the generated .mp3 file.
    Uses a cache based on the hash of the text and voice to avoid re-synthesizing identical requests.
    
    Args:
        text (str): The text to speak.
        voice (str): The edge-tts voice identifier.
        
    Returns:
        str: Path to the generated mp3 file.
    """
    if not text:
        return None

    voice, rate, pitch = get_voice_settings(emotion)
    # Create a unique filename based on content hash
    content_hash = hashlib.md5(f"{text}|{voice}|{rate}|{pitch}".encode("utf-8")).hexdigest()
    filename = f"{content_hash}.mp3"
    output_path = os.path.join(CACHE_DIR, filename)
    
    # Return cached if exists
    if os.path.exists(output_path):
        return output_path
        
    try:
        # edge-tts is async, so we run it in a loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_synthesize_async(text, output_path, voice, rate, pitch))
        loop.close()
        return output_path
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None

if __name__ == "__main__":
    # Test
    test_text = "Hello, I am EmoCare. How are you feeling today?"
    print(f"Synthesizing: {test_text}")
    path = synthesize_speech(test_text)
    print(f"Output saved to: {path}")

