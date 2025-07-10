import os
import gradio as gr
import tempfile
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

def generate_explanation(question, language):
    prompt = f"""
    You are a kind and patient teacher who excels at explaining concepts to young students who do not have access to good educational resources.
    ###Instructions:
    - Explain this question in {language} using simple language and easy analogy but make sure to be factual and not dreamy. Just explain straight facts but in a simpler manner for the students to understand.
    - If the question involves historical real life incidents be straightforward and specify the direct answer as a definition and don't make metaphors or give dreamy explanations.
    - Do not give an extended explanation involving unecessary comments. Be straightforward yet explanatory so that the student can easily understand the explanation.
    - The explanation should be at most 7 lines and not more than that. 
    ###Question: 
    {question}
"""
    response = model.generate_content(prompt)
    return response.text


def transcribe_audio(audio_bytes):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-IN"
    )
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else "Could not transcribe"

language_code_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml"
}

def process_input(text_input, audio_file, language):
    if text_input and text_input.strip():
        question = text_input.strip()
    elif audio_file:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        question = transcribe_audio(audio_bytes)
    else:
        return "Please enter or record a question.", "", None

    explanation = generate_explanation(question, language)

    
    tts = gTTS(text=explanation, lang=language_code_map[language])
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    return question, explanation, temp_audio.name

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Type your question (optional)"),
        gr.Audio(type="filepath", label="Or record your question (optional)"),
        gr.Dropdown(choices=list(language_code_map.keys()), label="Choose Language")
    ],
    outputs=[
        gr.Textbox(label="Final Question (Transcribed or Typed)"),
        gr.Textbox(label="Simple Explanation"),
        gr.Audio(label="Hear the Answer")
    ],
    title="Instant Knowledge Buddy",
    description="Ask by typing or speaking. Get a simple explanation with an analogy in your local language."
)

iface.launch()
