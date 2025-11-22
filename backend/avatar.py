import os

BASE_DIR = os.path.join("assets", "avatars")

# Map canonical emotion labels to available avatar files.
# Available avatars: neutral.png, happy.png, sad.png, angry.png, anxious.png
EMO2FILE = {
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "fear": "anxious.png",
    "disgust": "anxious.png",
    "surprised": "happy.png",
    "neutral": "neutral.png",
}

def avatar_for_emotion(emotion: str) -> str:
    """
    Returns the path to the avatar image for the given emotion.
    Expects canonical emotion labels: ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
    """
    emotion = (emotion or "neutral").lower()
    filename = EMO2FILE.get(emotion, "neutral.png")
    return os.path.join(BASE_DIR, filename)
