# Sahayak: AI-Powered Teaching Assistant
# Auto-detects textbook language and generates multi-grade worksheets
import gradio as gr
import base64
import requests
from PIL import Image
import io
import subprocess
import os
import tempfile
from langdetect import detect
from dotenv import load_dotenv

from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VISION_API_KEY = os.getenv("VISION_API_KEY")
USE_OFFLINE = False

# Language Mapping
LANG_CODE_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'ta': 'Tamil'
}
TESSERACT_LANGS = "eng+mar+hin+tam"

import markdown2
from xhtml2pdf import pisa

def is_online():
    return bool(GEMINI_API_KEY and VISION_API_KEY)

# --- OCR ---
def perform_ocr_online(image: Image.Image) -> str:
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "requests": [{
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        return result["responses"][0]["fullTextAnnotation"]["text"]
    except Exception as e:
        return f"[OCR Online Failed] {e}"

def perform_ocr_offline(image: Image.Image) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            image.save(temp_img.name)
            result = subprocess.run(
                ["tesseract", temp_img.name, "stdout", "-l", TESSERACT_LANGS],
                capture_output=True, text=True
            )
        return result.stdout.strip()
    except Exception as e:
        return f"[OCR Offline Failed] {e}"

# --- Language Detection ---
def detect_language(text: str) -> str:
    try:
        lang_code = detect(text)
        return LANG_CODE_MAP.get(lang_code, "English")
    except:
        return "English"

# --- LLM Generators ---
def generate_with_gemini(text: str, language: str) -> str:
    prompt = f"""
You are a multilingual teaching assistant. Based on the following textbook content, create four differentiated worksheets (Grade 1 to Grade 4) in {language}. Each worksheet should be age-appropriate, increase in difficulty, and include a variety of questions.

Content:
{text.strip()}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini Generation Failed] {e}"  

def generate_with_tinyllama(text: str, language: str) -> str:
    prompt = f"""
You are a multilingual teaching assistant. Based on the following textbook content, create four differentiated worksheets (Grade 1 to Grade 4) in {language}. Each worksheet should be age-appropriate, increase in difficulty, and include a variety of questions.

Content:
{text.strip()}
"""
    try:
        output = subprocess.run([
            "llama.cpp/main",  # Update this path
            "-m", "models/tinyllama-gguf/tinyllama.gguf",
            "-p", prompt,
            "--n-predict", "512"
        ], capture_output=True, text=True)
        return output.stdout.strip()
    except Exception as e:
        return f"[TinyLlama Failed] {e}"

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def worksheet_to_pdf(worksheet_text: str) -> str:
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    font_path = os.path.abspath("NotoSans-Regular.ttf")
    html = markdown2.markdown(worksheet_text)
    html_with_font = f"""
    <html>
    <head>
    <style>
    @font-face {{
        font-family: 'NotoSans';
        src: url('file://{font_path}');
    }}
    body {{
        font-family: 'NotoSans', sans-serif;
    }}
    </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    with open(temp_pdf.name, "wb") as pdf_file:
        pisa.CreatePDF(html_with_font, dest=pdf_file)
    return temp_pdf.name

# --- Final Pipeline ---
def smart_pipeline(image: Image.Image):
    global USE_OFFLINE
    USE_OFFLINE = not is_online()

    ocr_result = perform_ocr_offline(image) if USE_OFFLINE else perform_ocr_online(image)
    detected_lang = detect_language(ocr_result)
    worksheet = (
        generate_with_tinyllama(ocr_result, detected_lang)
        if USE_OFFLINE else
        generate_with_gemini(ocr_result, detected_lang)
    )

    mode = "📴 Offline Mode" if USE_OFFLINE else "🌐 Online Mode"
    result_text = f"{mode}\nDetected Language: {detected_lang}\n\n--- Extracted Text ---\n{ocr_result}\n\n--- Generated Worksheet ---\n{worksheet}"
    pdf_path = worksheet_to_pdf(worksheet)
    return result_text, pdf_path

# --- Gradio UI ---
demo = gr.Interface(
    fn=smart_pipeline,
    inputs=gr.Image(type="pil", label="Upload textbook page"),
    outputs=[gr.Textbox(label="Generated Worksheets"), gr.File(label="Download Worksheet as PDF")],
    title="Sahayak: Multi-Grade Worksheet Generator",
    description="Upload a textbook image and get differentiated worksheets for Grades 1–4. The assistant auto-detects language and works offline if needed.",
    theme="default"
)

demo.launch()
