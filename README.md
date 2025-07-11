# Sahayak 360 – AI Teaching Companion for Multilingual, Multi-Grade Classrooms

**Sahayak 360** is a unified AI-powered platform designed to empower teachers in low-resource, rural classrooms. It provides real-time assistance with **lesson planning**, **visual aid generation**, **multilingual explanations**, **differentiated worksheets**, and **hyperlocal storytelling** — all through **voice input**, in **local languages**, and with full **offline fallback**.

---

## Key Features

## Modular Agent Suite

| Module                          | Description |
|---------------------------------|-------------|
| **Weekly Lesson Planner**       | Voice-based scheduler generating structured lesson timelines + worksheets + calendar integration |
| **Visual Aids Generator**       | Converts voice/text prompts into chalkboard-style diagrams using Gemini + SDXL + Mermaid.js |
| **Differentiated Worksheets**   | Generates grade-wise assignments from a single textbook image using OCR + Gemini Vision |
| **Knowledge Explainer Bot**     | Simplifies complex queries into age-appropriate, analogy-rich responses with TTS + multilingual support |
| **Hyperlocal Content Generator**| Crafts culturally relevant stories, poems, games, and festival content in local languages |

---

## Voice + Multilingual Capabilities

- Accepts voice input in **Hindi**, **Marathi**, **Tamil**, and more  
- Outputs explanations, labels, and content in the **same language**  
- Optional **Text-to-Speech (TTS)** playback for auditory learning  
- **Local font rendering** (Noto Sans, Devanagari, Tamil, etc.)

---

## Offline-First Architecture

Sahayak 360 is built to run seamlessly in **low or zero-connectivity environments**:

- **TinyLlama + llama.cpp** for local LLM inference  
- **RAG (Retrieval-Augmented Generation)** over lightweight SQLite content cache  
- **Tesseract OCR** for image-based content extraction  
- Smart sync logic for updating cache when internet reconnects  

---

## Technology Stack

### Backend / AI
- Gemini 1.5 Pro  
- Text-Bison / PaLM API  
- TinyLlama (llama.cpp)  
- Vertex AI Speech-to-Text  
- Google Cloud Vision API / Tesseract OCR  
- Google Translate API / googletrans  

### Frontend
- Streamlit (web interface)  
- Flutter (mobile-first planned)  
- Gradio (for tool testing)  

### Tooling & Output
- Mermaid.js (flowcharts)  
- Hugging Face SDXL + BLIP (sketches + captions)  
- fpdf2 / reportlab (PDF exports)  
- Google Docs & Calendar APIs  

### Data Handling
- SQLite / JSON (offline content cache)  
- RAG pipeline for semantic search over local datasets  

---

## System Architecture

![Sahayak Architecture](https://github.com/user-attachments/assets/73f6a661-07ef-4b58-bd48-ed4554c2890c)

The system routes voice/text/image inputs through multilingual STT, OCR, or language detection. Based on context and connectivity, the appropriate agent is triggered (Gemini online or TinyLlama offline), followed by structured content generation using the JSON Tool Suite. Output flows into PDF/Calendar/TTS modules depending on user needs.

---

## Sample Use Cases

- A teacher speaks “Plan Grade 4 science next week” → Schedules week with daily activities, worksheets, and calendar events  
- Upload a Marathi textbook photo → Generates differentiated worksheets for Grades 1–4  
- Ask “Why do stars twinkle?” in Hindi → Receives grade-wise, analogy-rich explanations + TTS playback  
- Request a poem about seasons for Class 2 → Returns a locally themed story in Devanagari script  
- Say “Draw the water cycle” → Visual aid + sketch + step-by-step chalkboard instructions  

---

## Real-World Impact

- Empowers rural and multi-grade teachers with instant, personalized support  
- Works even in **no-internet zones** — supports low-end Android devices  
- Promotes **regional languages and cultural relevance** in classroom content  
- Saves hours of lesson preparation while improving student engagement  

---
##Note: This is an early-stage implementation of Sahayak 360. The current submission showcases our modular architecture, key workflows, and offline-first capabilities — while the full-featured prototype is under active development.
