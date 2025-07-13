import gradio as gr
import os
import json
import datetime
from google.generativeai.generative_models import GenerativeModel
from dotenv import load_dotenv
import google.generativeai

# Load API keys
a=load_dotenv()
print(a)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup Gemini Model
model = GenerativeModel("gemini-1.5-pro")

# Prompt template
def generate_prompt(user_input, language):
    today = datetime.date.today()
    lang_instruction = f"The teacher is speaking in {language}."
    return f"""
You are an AI lesson planner assistant for multi-grade classrooms in rural India.

Today's date: {today.strftime('%B %d, %Y')}

{lang_instruction}
User instruction: "{user_input}"

Generate a culturally-aware weekly lesson plan (Monday–Saturday) in JSON format.
Each day must include:
- Grade
- Topic
- Activity
- Cultural/festival notes if applicable

Use short, simple phrasing appropriate for that language and context.
Output must be in the same language.
"""

# Core function
def create_lesson_plan(user_input, language):
    prompt = generate_prompt(user_input, language)
    try:
        response = model.generate_content(prompt)
        content = response.text

        # Try to parse JSON if available
        try:
            plan_json = json.loads(content)
            formatted = json.dumps(plan_json, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            formatted = content  # fallback

        return formatted
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(title="Sahayak – Weekly Planner (Multilingual)") as app:
    gr.Markdown("# 🌍📅 Sahayak – Weekly Planner")
    gr.Markdown("Enter your request below to generate a weekly teaching plan in your preferred language.")

    with gr.Row():
        text_input = gr.Textbox(label="Lesson Planning Request", placeholder="e.g. कक्षा 3 और 4 के लिए अगले हफ्ते योजना बनाएं, विषय: पानी और वायु")
        language_dropdown = gr.Dropdown(
            label="Select Language of Input/Output",
            choices=["English", "Hindi", "Tamil", "Marathi", "Telugu", "Bengali"],
            value="Hindi"
        )

    output = gr.Textbox(label="Generated Weekly Plan", lines=20)
    generate_btn = gr.Button("Generate Plan")

    generate_btn.click(fn=create_lesson_plan, inputs=[text_input, language_dropdown], outputs=output)

app.launch()
