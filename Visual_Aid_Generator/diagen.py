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


load_dotenv()
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

def generate_gemini_image(prompt: str) -> Image.Image:
    """Call Gemini 2.0 Flash Experimental API to get base64 inline image."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    }
    headers = {"Content-Type": "application/json"}

    res = requests.post(url, json=payload, headers=headers)
    res.raise_for_status()
    data = res.json()

    parts = data.get("candidates", [])[0]["content"]["parts"]
    for part in parts:
        if "inlineData" in part:
            image_base64 = part["inlineData"]["data"]
            image_bytes = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_bytes))

    raise RuntimeError("No image found in Gemini response.")


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
use_image = st.checkbox("Generate AI sketch using Gemini Flash?")
use_mermaid = st.checkbox("Generate structured diagram using Mermaid.js?")

if st.button("Generate Drawing Instructions") and prompt:
    with st.spinner("Generating drawing instructions via Gemini..."):
        try:
            system_prompt = """
You are an assistant that generates simple, numbered blackboard drawing steps based on a topic description.
###Instructions:
- **DO NOT GIVE LONG OR DIFFICULT TO UNDERSTAND DRAWING STEPS**
- **Also specify in the instructions to make sure the diagram is in black and white so that it can be easily replicated on a blackboard.**
- Mention that the diagram should be labelled appropriately. For example include in the line before generating the steps "and also label them with the following:".
- These instructions will be sent as the input prompt for image generation.
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
                st.subheader("AI-Generated Sketch (via Gemini Flash Experimental)")
                try:
                    with st.spinner("Generating sketch via Gemini..."):
                        image = generate_gemini_image(drawing_instructions)
                        st.image(image, caption="Gemini Image Sketch", use_column_width=True)

                    with st.spinner("Captioning the image..."):
                        caption = generate_image_caption(image)
                        st.subheader("Auto-Generated Caption:")
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
