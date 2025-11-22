from collections import Counter

# Central mapping to canonical labels
# Target set: ["happy", "sad", "angry", "surprised", "neutral", "fear", "disgust"]
EMOTION_MAPPING = {
    # Text (DistilRoBERTa)
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprised",
    "neutral": "neutral",
    
    # Audio (SpeechBrain IEMOCAP)
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
    "neu": "neutral",
    "exc": "happy", # Excited often maps to happy/high arousal
    
    # Image (ResNet / FER2013)
    "happy": "happy",
    "angry": "angry",
    "surprised": "surprised", # If model returns 'surprised'
    "surprise": "surprised",  # If model returns 'surprise'
    # 'fear' and 'disgust' and 'neutral' map to themselves
}

# Reliability-based weights
WEIGHTS = {
    "text": 0.7,   # High reliability
    "image": 0.15,  # Low reliability
    "audio": 0.15   # Low reliability (often overconfident neutral)
}

AUDIO_SCORE_CAP = 0.9
STRONG_TEXT_THRESHOLD = 0.9

def normalize_label(label: str) -> str:
    """Normalizes a raw label to the canonical set."""
    if not label:
        return "neutral"
    label = label.lower()
    return EMOTION_MAPPING.get(label, label) # Default to itself if not in map

def fuse_emotions(results: dict) -> dict:
    """
    Fuses emotion predictions using weighted average and rule-based gating.
    """
    scores = Counter()
    prepared = {}
    
    # Normalize / sanitize modality outputs first (handles capping + list labels)
    for modality in ["text", "image", "audio"]:
        res = results.get(modality)
        if not res:
            continue
        
        raw_label = res.get("label")
        if isinstance(raw_label, list):
            raw_label = raw_label[0] if raw_label else None
        
        if not raw_label or raw_label == "error":
            continue
        
        canonical_label = normalize_label(raw_label)
        score = res.get("score", 0.0) or 0.0
        
        # Cap audio confidence before any downstream use (and reflect in debug)
        if modality == "audio" and score > AUDIO_SCORE_CAP:
            score = AUDIO_SCORE_CAP
            res["score"] = AUDIO_SCORE_CAP
        
        prepared[modality] = {"label": canonical_label, "score": score}
    
    r_text = prepared.get("text")
    
    # 1. Check for Strong Text (Gating Rule)
    if r_text:
        text_label = r_text["label"]
        text_score = r_text["score"]
        
        if text_label != "neutral" and text_score > STRONG_TEXT_THRESHOLD:
            return {
                "label": text_label,
                "scores": {text_label: text_score, "rule": "strong_text_override"}
            }

    # 2. Weighted Fusion
    for modality, data in prepared.items():
        label = data["label"]
        score = data["score"]
        weighted_score = score * WEIGHTS.get(modality, 0.1)
        scores[label] += weighted_score

    if not scores:
        return {"label": "neutral", "scores": {}}

    fused_label, fused_score = scores.most_common(1)[0]
    fused_scores = dict(scores)
    
    # Confidence threshold: default to neutral if the best weighted score is weak
    if fused_score < 0.3:
        return {
            "label": "neutral",
            "scores": fused_scores | {"rule": "low_confidence_neutral"}
        }
    
    return {
        "label": fused_label,
        "scores": fused_scores
    }

if __name__ == "__main__":
    # Simple test
    test_results = {
        "text": {"label": "joy", "score": 0.9},      # Should map to happy
        "audio": {"label": ["hap"], "score": 0.6},   # Should map to happy
        "image": {"label": "surprise", "score": 0.7},# Should map to surprised
    }
    print(f"Input: {test_results}")
    print(f"Fused: {fuse_emotions(test_results)}")
