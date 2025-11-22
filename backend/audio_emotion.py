import torch
import torchaudio
import soundfile as sf
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="inspect") # Filter inspect module warnings if needed, though hard to target specifically


# Polyfill for torchaudio > 2.1 where list_audio_backends is removed
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

# Updated import path for SpeechBrain 1.0+
from speechbrain.inference.classifiers import EncoderClassifier
from backend.config import get_device

# Use the device from config
device = get_device()
run_opts = {"device": device}

MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

class CustomEncoderClassifier(EncoderClassifier):
    def classify_batch(self, wavs, wav_lens=None):
        """Custom classify batch that manually runs the model components
        since the interface expects a compute_features module that isn't in the YAML."""
        
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Manually run the forward pass based on hyperparams.yaml structure
        
        # 1. Wav2Vec2
        # The SpeechBrain Wav2Vec2 wrapper returns the raw output from the HF model.
        # Typically this is a tensor (last_hidden_state) or an object with attributes.
        # Inspecting source: it returns just the tensor if not configured otherwise, 
        # or the HF output object.
        
        outputs = self.mods.wav2vec2(wavs, wav_lens)
        
        # Handle return type: if it's a Tensor, use it directly.
        # If it's a HF ModelOutput (e.g. BaseModelOutput), access last_hidden_state.
        if isinstance(outputs, torch.Tensor):
            feats = outputs
        elif hasattr(outputs, 'last_hidden_state'):
            feats = outputs.last_hidden_state
        else:
            # Fallback: try subscripting (tuple) or just assume it's the tensor
            feats = outputs[0] if isinstance(outputs, tuple) else outputs

        # 2. Pooling
        feats = self.mods.avg_pool(feats, wav_lens)

        # 3. Output MLP
        outputs = self.mods.output_mlp(feats)
        
        # 4. Softmax
        outputs = self.hparams.softmax(outputs)

        # Decode
        score, index = torch.max(outputs, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        
        return outputs, score, index, text_lab

# Load model once
try:
    # savedir is where the model will be downloaded
    classifier = CustomEncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts=run_opts,
        savedir="pretrained_models/speechbrain_emotion"
    )
    # Explicitly tell the label encoder to expect 4 classes (neu, ang, hap, sad)
    # This silences the "CategoricalEncoder.expect_len was never called" warning.
    # We verify this count by inspecting pretrained_models/speechbrain_emotion/label_encoder.ckpt
    if hasattr(classifier.hparams, "label_encoder"):
        classifier.hparams.label_encoder.ignore_len()
        
except Exception as e:
    print(f"Error loading SpeechBrain model {MODEL_SOURCE}: {e}")
    raise e

def analyze_audio(wav_path: str) -> dict:
    """
    Analyzes the emotion of the audio file at the given path.
    The model expects 16kHz audio; this function resamples if necessary.
    
    Args:
        wav_path (str): Path to the .wav file.
        
    Returns:
        dict: {'label': <emotion>, 'score': <probability>}
    """
    # Load audio file with soundfile to avoid torchcodec dependency
    # soundfile returns shape [time] or [time, channels]
    try:
        samples, fs = sf.read(wav_path, dtype="float32", always_2d=True)
    except Exception as e:
         return {"label": "error", "score": 0.0, "details": str(e)}

    # soundfile returns [time, channels]; convert to torch [channels, time]
    samples = torch.from_numpy(samples.T)  # [channels, time]

    signal = samples

    # Convert to mono if stereo (average channels)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed (standard for this model)
    target_sample_rate = 16000
    if fs != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sample_rate)
        signal = resampler(signal)
    
    # Move signal to device
    signal = signal.to(device)
    
    with torch.no_grad():
        # Returns: out_prob, score, index, text_lab
        out_prob, score, index, text_lab = classifier.classify_batch(signal)
    
    # Handle score extraction
    if isinstance(score, torch.Tensor):
        # Flatten if needed
        final_score = score.view(-1)[0].item()
    else:
        final_score = score
        
    # Handle label extraction and unnesting
    final_label = text_lab[0]
    
    # Recursively unwrap if it's a list (e.g. [['hap']] or ['hap'])
    while isinstance(final_label, list):
        if len(final_label) > 0:
            final_label = final_label[0]
        else:
            final_label = "neutral"
            break
            
    return {
        "label": str(final_label), # Ensure strictly string
        "score": final_score
    }

if __name__ == "__main__":
    print(analyze_audio("test_audio.wav"))
