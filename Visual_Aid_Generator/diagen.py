import streamlit as st
import os
import requests
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Sahayak - Drawing Generator", layout="centered")
st.title("Sahayak: Blackboard Drawing Assistant")


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()


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

def generate_image_caption(image: Image.Image) -> str:
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_mermaid_code(prompt: str) -> str:
    mermaid_prompt = f"""
You are an assistant that generates mermaid.js diagrams based on a user's diagram request.
Generate a Mermaid flowchart that clearly shows labeled nodes and arrows for:
"{prompt}"
Return only the mermaid code block.
"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(mermaid_prompt)
    return response.text.strip()

prompt = st.text_input("Enter the diagram topic or description:", "Draw the water cycle with sun, ocean, clouds, rain, and arrows")
use_image = st.checkbox("Generate AI sketch using Hugging Face SDXL?")
use_mermaid = st.checkbox("Generate structured diagram using Mermaid.js?")

if st.button("Generate Drawing Instructions") and prompt:
    with st.spinner("Generating drawing instructions via Gemini..."):
        try:
            system_prompt = """
You are an assistant that generates simple, numbered blackboard drawing steps based on a topic description.
###Instructions:
- **DO NOT GIVE LONG OR DIFFICULT TO UNDERSTAND DRAWING STEPS**
- **Also specify in the instructions to make sure the diagram is in black and white so that it can be easily replicated on a blackboard.**
- These instructions will be sent as the input prompt for Stavle Diffusion's image generation base model.
- Do not exceed 5 steps.
- Mention the shapes of each element in the diagram.
- Make sure the instructions are self explanatory.

"""
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content([system_prompt, prompt])
            drawing_instructions = response.text.strip()

            st.subheader("Drawing Instructions:")
            st.code(drawing_instructions, language="markdown")

            if use_image:
                st.subheader("AI-Generated Sketch (Stable Diffusion XL)")
                try:
                    with st.spinner("Generating sketch using Hugging Face SDXL..."):
                        image = generate_sdxl_image(prompt)
                        st.image(image, caption="SDXL Sketch", use_column_width=True)

                        with st.spinner("Captioning the image..."):
                            caption = generate_image_caption(image)
                            st.subheader("📟 Auto-Generated Caption:")
                            st.markdown(f"**{caption}**")
                except Exception as e:
                    st.error(f"Image generation or captioning failed: {str(e)}")

            if use_mermaid:
                st.subheader("Mermaid.js Code for Diagram")
                try:
                    with st.spinner("Generating Mermaid diagram code..."):
                        mermaid_code = generate_mermaid_code(prompt)
                        st.code(mermaid_code, language="markdown")
                except Exception as e:
                    st.error(f"Mermaid code generation failed: {str(e)}")

            st.success("Ready to draw or print!")

        except Exception as e:
            st.error(f"Gemini generation failed: {str(e)}")

st.markdown("---")
