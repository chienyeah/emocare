import gradio as gr
import numpy as np
import soundfile as sf
import os
import tempfile
from PIL import Image as PILImage
import traceback
import base64

# Backend imports
from backend.text_emotion import analyze_text
from backend.audio_emotion import analyze_audio
from backend.image_emotion import analyze_image
from backend.fusion import fuse_emotions
from backend.comfort_llm import generate_comfort
from backend.avatar import avatar_for_emotion
from backend.tts_service import synthesize_speech
from backend.stt_service import transcribe_audio_file

import shutil

def save_audio_temp(audio):
    """
    Saves gradio audio input (fs, numpy_array) to a temporary wav file.
    Returns the path to the temp file.
    """
    if audio is None:
        return None
        
    sample_rate, data = audio
    
    # Defensive check for empty data which can cause errors
    if len(data) == 0:
        return None
    
    # Create a temp file
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    # Write to wav
    try:
        sf.write(path, data, sample_rate)
    except Exception as e:
        print(f"Error writing temp audio: {e}")
        try:
            os.remove(path)
        except:
            pass
        return None
        
    return path

def clear_voice_cache():
    """Clears all files in the voice_cache directory."""
    cache_dir = os.path.join("assets", "voice_cache")
    if os.path.exists(cache_dir):
        try:
            # Option 1: Delete all files inside
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            print("Voice cache cleared.")
        except Exception as e:
            print(f"Error clearing voice cache: {e}")

def append_transcription(audio, current_text):
    """
    Transcribes recorded audio and appends it to the existing textbox content.
    """
    existing_text = current_text or ""
    if audio is None:
        return existing_text

    # Save temp for transcription
    temp_path = save_audio_temp(audio)
    if not temp_path:
        return existing_text

    try:
        transcribed = transcribe_audio_file(temp_path)
    except Exception as e:
        print(f"Transcription error: {e}")
        transcribed = ""
    finally:
        try:
            os.remove(temp_path)
        except:
            pass

    if not transcribed:
        return existing_text

    # Append
    if existing_text:
        combined = (existing_text + " " + transcribed.strip()).strip()
    else:
        combined = transcribed.strip()
        
    return combined

def convert_history_for_llm(history):
    """
    Converts Gradio Chatbot history (list of dicts with role/content/raw_content) into a list of (user, assistant) tuples
    for compatibility with the LLM context format. Uses the raw_content field if present, which strips any HTML/audio embeds.
    """
    if not history:
        return []
    pairs = []
    current_user = None
    for msg in history:
        role = msg.get("role")
        content = msg.get("raw_content", msg.get("content", ""))
        if role == "user":
            current_user = content
        elif role == "assistant":
            if current_user is None:
                current_user = ""
            pairs.append((current_user, content))
            current_user = None
    return pairs


def emocare_step(user_text, audio, image, history):
    """
    Main processing step for EmoCare.
    """
    if history is None:
        history = []

    # Default fallback
    if not user_text and audio is None and image is None:
        return history, history, "neutral", {"info": "No input provided"}, avatar_for_emotion("neutral"), "", None

    try:
        results = {"text": None, "audio": None, "image": None}
        
        # Initialize context_text with existing user_text
        context_text = user_text if user_text else ""

        # 1. Audio Analysis (Transcribe removed here to avoid duplication if UI does it)
        # But we still need audio for emotion analysis
        if audio is not None:
            temp_wav_path = save_audio_temp(audio)
            if temp_wav_path:
                # Analyze emotion
                results["audio"] = analyze_audio(temp_wav_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_wav_path)
                except:
                    pass
        
        # 2. Text Analysis
        if context_text and context_text.strip():
            results["text"] = analyze_text(context_text)
        
        if not context_text:
             context_text = "(User provided non-text/speech input)"

        # 3. Image Analysis
        if image is not None:
            # Gradio image input might be numpy array or PIL
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            results["image"] = analyze_image(pil_image)
            
        # 4. Fusion
        fusion_result = fuse_emotions(results)
        fused_label = fusion_result["label"]
        fused_scores = fusion_result["scores"]
        
        # 5. Generate Response
        llm_history = convert_history_for_llm(history)
        reply = generate_comfort(context_text, fused_label, llm_history)
        
        # 6. Synthesize Speech (matching fused emotion)
        audio_response_path = synthesize_speech(reply, fused_label)
        
        # 7. Update History
        if audio_response_path:
            abs_path = os.path.abspath(audio_response_path)
            try:
                with open(abs_path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                audio_html = f'<br><audio controls src="data:audio/mpeg;base64,{audio_base64}"></audio>'
            except Exception:
                audio_html = ""
            history.append({"role": "user", "content": context_text, "raw_content": context_text})
            history.append({"role": "assistant", "content": reply + audio_html, "raw_content": reply})
        else:
            history.append({"role": "user", "content": context_text, "raw_content": context_text})
            history.append({"role": "assistant", "content": reply, "raw_content": reply})
        
        # 8. Get Avatar
        avatar_path = avatar_for_emotion(fused_label)
        
        # 9. Debug Info
        debug_info = {
            "modality_results": results,
            "fusion_scores": fused_scores
        }
        
        # Return empty string to clear input box
        return history, history, fused_label.title(), debug_info, avatar_path, "", audio_response_path
        
    except Exception as e:
        print(f"Error in emocare_step: {e}")
        traceback.print_exc()
        # Fallback in case of error
        error_reply = "I'm sorry, I had a little trouble understanding that. Could you try again?"
        user_raw = user_text if user_text else "..."
        history.append({"role": "user", "content": user_raw, "raw_content": user_raw})
        history.append({"role": "assistant", "content": error_reply, "raw_content": error_reply})
        # Return a safe fallback tuple
        return history, history, "Error", {"error": str(e)}, avatar_for_emotion("neutral"), "", None

# Custom CSS
theme_css = """
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #fdfbf7; color: #4a4a4a; }
gradio-app { background-color: #fdfbf7 !important; }
header { margin-bottom: 2rem; text-align: center; }
h1 { font-family: 'Garamond', serif; color: #2c3e50; font-size: 2.5em; margin-bottom: 0.2em; }
.subtitle { font-size: 1.1em; color: #7f8c8d; font-style: italic; }
.contain { max-width: 1100px !important; margin: 0 auto !important; }
.chatbot { height: 450px !important; border-radius: 12px; border: 1px solid #e0e0e0; background-color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.input-box { border-radius: 8px !important; }
.avatar-card { background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center; border: 1px solid #eee; }
.avatar-img { border-radius: 50%; width: 150px; height: 150px; object-fit: cover; margin: 0 auto 15px auto; display: block; border: 4px solid #f0f0f0; }
.emotion-label { font-size: 1.5em; font-weight: bold; color: #34495e; margin-bottom: 10px; display: block; }
.disclaimer { font-size: 0.8em; color: #95a5a6; margin-top: 20px; text-align: center; }
"""

with gr.Blocks(css=theme_css, title="EmoCare") as demo:
    
    # State
    chat_history = gr.State([])
    
    # Header
    with gr.Row(elem_id="header"):
        with gr.Column():
            gr.Markdown(
                """
                # EmoCare
                <div class="subtitle">A multimodal emotion comfort companion.</div>
                """
            )

    with gr.Row():
        # Left Column: Conversation & Inputs
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_classes="chatbot",
                show_label=False,
                avatar_images=(None, "assets/avatars/neutral.png"), 
                type="messages"
            )
            
            # Audio output for TTS response
            tts_audio = gr.Audio(label="Companion Voice", interactive=False, autoplay=True, visible=False)
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type how you feel...",
                    lines=2,
                    scale=4,
                    elem_classes="input-box"
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    label="Or speak how you feel", 
                    type="numpy"
                )
                image_input = gr.Image(
                    sources=["webcam"], 
                    label="Optional: show your face", 
                    type="numpy"
                )
            
            # Wiring transcription event - triggers when recording stops
            # Use .change instead of .stop_recording? Or both?
            # When deleting, .change fires with None.
            # When stopping recording, .stop_recording fires.
            # We use stop_recording for transcription.
            
            audio_input.stop_recording(
                fn=append_transcription,
                inputs=[audio_input, user_input],
                outputs=[user_input]
            )
            
            gr.Markdown("*Camera and microphone are optional. All sensitive data stays on your device.*", elem_classes="disclaimer")

        # Right Column: Emotion & Avatar Panel
        with gr.Column(scale=3):
            with gr.Group(elem_classes="avatar-card"):
                # Avatar Display
                avatar_display = gr.Image(
                    value=avatar_for_emotion("neutral"), 
                    label="Companion", 
                    show_label=False,
                    show_download_button=False,
                    container=False,
                    elem_classes="avatar-img",
                    interactive=False
                )
                
                # Detected Emotion Label
                emotion_label = gr.Label(
                    value="Neutral", 
                    label="Detected Emotion",
                    show_label=True,
                    num_top_classes=1
                )
                
                # Debug Info
                with gr.Accordion("Analysis Details", open=False):
                    debug_output = gr.JSON(label="Modality Scores")

            gr.Markdown(
                "EmoCare is not a clinical tool; it’s here for gentle support.",
                elem_classes="disclaimer"
            )

    # Event Wiring
    inputs = [user_input, audio_input, image_input, chat_history]
    outputs = [chatbot, chat_history, emotion_label, debug_output, avatar_display, user_input, tts_audio]
    
    send_btn.click(
        fn=emocare_step,
        inputs=inputs,
        outputs=outputs
    )
    
    user_input.submit(
        fn=emocare_step,
        inputs=inputs,
        outputs=outputs
    )

    # Clear Logic
    def clear_state():
        # Clear voice cache when conversation is cleared
        clear_voice_cache()
        return [], [], "Neutral", None, avatar_for_emotion("neutral"), "", None

    chatbot.clear(
        fn=clear_state,
        inputs=[],
        outputs=[chat_history, chatbot, emotion_label, debug_output, avatar_display, user_input, tts_audio]
    )

if __name__ == "__main__":
    demo.launch()
