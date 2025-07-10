import streamlit as st
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit UI setup
st.set_page_config(page_title="Sahayak - Drawing Generator", layout="centered")
st.title("Sahayak: Blackboard Drawing Assistant")

# Load BLIP model for captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()


# Generate image from SDXL
def generate_sdxl_image(prompt: str):
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise RuntimeError(f"HF API failed: {response.status_code} - {response.text}")


# Generate caption using BLIP
def generate_image_caption(image: Image.Image) -> str:
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# UI Inputs
prompt = st.text_input("Enter the diagram topic or description:", "Draw the water cycle with sun, ocean, clouds, rain, and arrows")

use_image = st.checkbox("Generate AI sketch using Hugging Face SDXL?")

# Main logic
if st.button("Generate Drawing Instructions") and prompt:
    with st.spinner("Generating drawing instructions via Gemini..."):
        try:
            system_prompt = """
You are an assistant that generates simple, numbered blackboard drawing steps based on a topic description. Make the instructions simple and specific enough for Stability image generator to understand properly.
Only return the steps, no explanation or metadata. Also make sure that you mention that the image generation steps are for a blackboard drawing with labels. Also generate a labelled line diagram of the image in the output with proper
"""
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content([system_prompt, prompt])
            drawing_instructions = response.text.strip()

            st.subheader("📋 Drawing Instructions:")
            st.code(drawing_instructions, language="markdown")

            

            if use_image:
                st.subheader("🖼️ AI-Generated Sketch (Stable Diffusion XL)")
                try:
                    with st.spinner("Generating sketch using Hugging Face SDXL..."):
                        image = generate_sdxl_image(prompt)
                        st.image(image, caption="SDXL Sketch", use_column_width=True)

                        with st.spinner("Captioning the image..."):
                            caption = generate_image_caption(image)
                            st.subheader("🧾 Auto-Generated Caption:")
                            st.markdown(f"**{caption}**")
                except Exception as e:
                    st.error(f"Image generation or captioning failed: {str(e)}")

            st.success("Ready to draw or print!")

        except Exception as e:
            st.error(f"Gemini generation failed: {str(e)}")

st.markdown("---")
