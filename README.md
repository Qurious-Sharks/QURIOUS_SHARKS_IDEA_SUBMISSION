# 🧠 Sahayak 360 – AI Teaching Companion for Multilingual, Multi-Grade Classrooms

Sahayak 360 is a unified AI-powered platform designed to empower teachers in low-resource, rural classrooms. It provides real-time assistance with **lesson planning**, **visual aid generation**, **multilingual explanations**, **differentiated worksheets**, and **hyperlocal storytelling** — all through **voice input**, in **local languages**, and with full **offline fallback**.

---

## 🚀 Key Features

### 🧩 Modular Agent Suite
| Module                          | Description |
|---------------------------------|-------------|
| 📅 **Weekly Lesson Planner**       | Voice-based scheduler generating structured lesson timelines + worksheets + calendar integration |
| 🖼️ **Visual Aids Generator**       | Converts voice/text prompts into chalkboard-style diagrams using Gemini + SDXL + Mermaid.js |
| 📚 **Differentiated Worksheets**   | Generates grade-wise assignments from a single textbook image with OCR + Gemini Vision |
| 🧠 **Knowledge Explainer Bot**     | Simplifies complex queries into age-appropriate, analogy-rich responses with TTS + multilingual support |
| 🏡 **Hyperlocal Content Generator**| Crafts culturally relevant stories, poems, games, and festival content in local languages |

---

## 🌐 Voice + Multilingual Capabilities

- Accepts voice input in **Hindi**, **Marathi**, **Tamil**, and more  
- Outputs explanations, labels, and content in the **same language**  
- Optional **Text-to-Speech** playback for auditory learning  
- **Local font rendering** (Noto Sans, Devanagari, etc.)

---

## 📶 Offline-First Architecture

Sahayak 360 is built to run seamlessly in **low or zero-connectivity environments**:

- 📦 **50MB Offline Mode**
- 🧠 **TinyLlama + llama.cpp** for local LLM inference
- 🔎 **RAG (Retrieval-Augmented Generation)** over SQLite cache
- 📷 **Tesseract OCR** for offline image processing
- 🔄 Smart sync logic for updating cache when reconnected

---

## 🛠️ Technology Stack

### Backend / AI
- [Gemini 1.5 Pro](https://ai.google.dev)
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
- HuggingFace SDXL + BLIP (sketch + captioning)
- fpdf2 / reportlab (PDF exports)
- Google Docs/Calendar API

### Data Handling
- SQLite / JSON (offline cache)
- RAG pipeline for semantic search over local data

---

## 🏗️ Architecture Overview

```mermaid
graph TD
  A[User Input: Voice/Text] --> B[STT / OCR / Lang Detection]
  B --> C[Gemini / TinyLlama (Offline)]
  C --> D[Tool Suite: Planner, Sketcher, StoryGen, Explainer]
  D --> E[Mermaid, SDXL, PDF Export, Calendar API]
  C --> F[RAG Cache (50MB)]
  F --> D
```

