from dotenv import load_dotenv
import os
import json
import gradio as gr
import google.generativeai as genai
from google.cloud import speech

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# --- Tool JSON Schemas ---
generate_local_story_json = {
    "name": "generate_local_story",
    "description": "Generate culturally relevant educational story",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "grade_level": {"type": "string"},
            "tone": {"type": "string", "enum": ["funny", "serious", "inspiring", "adventurous"]},
            "region": {"type": "string"}
        },
        "required": ["topic", "language", "grade_level", "region"],
        "additionalProperties": False
    }
}

generate_local_poem_json = {
    "name": "generate_local_poem",
    "description": "Generate culturally relevant educational poem",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "grade_level": {"type": "string"},
            "poem_type": {"type": "string", "enum": ["rhyming", "haiku", "free verse", "limerick"]}
        },
        "required": ["topic", "language", "grade_level"],
        "additionalProperties": False
    }
}

generate_local_dialogue_json = {
    "name": "generate_local_dialogue",
    "description": "Generate culturally relevant dialogue",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "characters": {"type": "string"},
            "context": {"type": "string"}
        },
        "required": ["topic", "language", "characters", "context"],
        "additionalProperties": False
    }
}

generate_local_proverb_json = {
    "name": "generate_local_proverb",
    "description": "Generate culturally relevant proverb",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"}
        },
        "required": ["topic", "language"],
        "additionalProperties": False
    }
}

generate_local_riddles_json = {
    "name": "generate_local_riddles",
    "description": "Generate culturally relevant riddles",
    "parameters": {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "topic": {"type": "string"},
            "difficulty": {"type": "string"}
        },
        "required": ["language", "topic", "difficulty"],
        "additionalProperties": False
    }
}

generate_local_song_json = {
    "name": "generate_local_song",
    "description": "Generate culturally relevant song",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "melody_type": {"type": "string"}
        },
        "required": ["topic", "language", "melody_type"],
        "additionalProperties": False
    }
}

generate_local_roleplay_json = {
    "name": "generate_local_roleplay",
    "description": "Generate culturally relevant roleplay scenario",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "characters": {"type": "string"},
            "scenario": {"type": "string"}
        },
        "required": ["topic", "language", "characters", "scenario"],
        "additionalProperties": False
    }
}



generate_local_quiz_json = {
    "name": "generate_local_quiz",
    "description": "Generate culturally relevant quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "grade_level": {"type": "string"},
            "question_type": {"type": "string", "enum": ["MCQ", "true/false", "short answer"]},
            "num_questions": {"type": "integer", "minimum": 1, "maximum": 10}
        },
        "required": ["topic", "language", "grade_level"],
        "additionalProperties": False
    }
}

generate_local_game_json = {
    "name": "generate_local_game",
    "description": "Generate culturally relevant educational game",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "language": {"type": "string"},
            "materials": {"type": "string"},
            "duration": {"type": "integer", "minimum": 5, "maximum": 60}
        },
        "required": ["topic", "language"],
        "additionalProperties": False
    }
}

generate_local_festival_content_json = {
    "name": "generate_local_festival_content",
    "description": "Generate festival-specific educational content",
    "parameters": {
        "type": "object",
        "properties": {
            "festival": {"type": "string"},
            "language": {"type": "string"},
            "grade_level": {"type": "string"}
        },
        "required": ["festival", "language"],
        "additionalProperties": False
    }
}

generate_local_case_study_json = {
    "name": "generate_local_case_study",
    "description": "Generate region-specific educational case study",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "region": {"type": "string"},
            "language": {"type": "string"},
            "grade_level": {"type": "string"}
        },
        "required": ["topic", "region", "language"],
        "additionalProperties": False
    }
}



record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record a question the AI couldn't answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}



tools = [
    {"type": "function", "function": generate_local_story_json},
    {"type": "function", "function": generate_local_poem_json},
    {"type": "function", "function": generate_local_dialogue_json},
    {"type": "function", "function": generate_local_proverb_json},
    {"type": "function", "function": generate_local_riddles_json},
    {"type": "function", "function": generate_local_song_json},
    {"type": "function", "function": generate_local_roleplay_json},
    {"type": "function", "function": generate_local_quiz_json},
    {"type": "function", "function": generate_local_game_json},
    {"type": "function", "function": generate_local_festival_content_json},
    {"type": "function", "function": generate_local_case_study_json},
    {"type": "function", "function": record_unknown_question_json},
]

def generate_local_story(topic, language, grade_level, tone, region):
    prompt = (
        f"Create a {tone} story in {language} for {grade_level} students about {topic}. "
        f"Set it in {region} with local landmarks, cultural elements, and traditional values. "
        "Include characters with local names and incorporate regional festivals or traditions. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"story": call_gemini(prompt)}

def generate_local_poem(topic, language, grade_level, poem_type):
    prompt = (
        f"Create a {poem_type} poem in {language} for {grade_level} students about {topic}. "
        "Use simple language with rhyme and rhythm. Include local cultural references and natural elements. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"poem": call_gemini(prompt)}

def generate_local_dialogue(topic, language, characters, context):
    prompt = (
        f"Create a short dialogue in {language} between {characters} about {topic}. "
        f"Context: {context}. Use natural speech patterns and local expressions. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"dialogue": call_gemini(prompt)}

def generate_local_proverb(topic, language):
    prompt = (
        f"Create an original proverb in {language} about {topic}. "
        "Make it wise, concise, and culturally appropriate with a moral lesson. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"proverb": call_gemini(prompt)}

def generate_local_riddles(language, topic, difficulty):
    prompt = (
        f"Create 3 {difficulty} riddles in {language} about {topic}. "
        "Include answers. Make them culturally relevant with local references. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"riddles": call_gemini(prompt)}

def generate_local_song(topic, language, melody_type):
    prompt = (
        f"Create a short children's song in {language} about {topic}. "
        f"Style: {melody_type}. Include a simple chorus and verses. "
        "Use local musical elements and repetitive structure for easy learning. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"song": call_gemini(prompt)}

def generate_local_roleplay(topic, language, characters, scenario):
    prompt = (
        f"Create a roleplay scenario in {language} about {topic} for students. "
        f"Characters: {characters}. Scenario: {scenario}. "
        "Include dialogue, actions, and learning objectives. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"roleplay": call_gemini(prompt)}




def generate_local_quiz(topic, language, grade_level, question_type, num_questions):
    prompt = (
        f"Create {num_questions} {question_type} questions in {language} about {topic} "
        f"for {grade_level} students. Include answers. Use local examples and contexts. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"quiz": call_gemini(prompt)}

def generate_local_game(topic, language, materials, duration):
    prompt = (
        f"Create a {duration}-minute educational game in {language} about {topic}. "
        f"Materials available: {materials}. Include rules, setup, and learning objectives. "
        "Use traditional game formats with local cultural elements. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"game": call_gemini(prompt)}

def generate_local_festival_content(festival, language, grade_level):
    prompt = (
        f"Create educational content in {language} about {festival} for {grade_level} students. "
        "Include its significance, traditions, related activities, and a simple craft idea. "
        "Connect to local practices and values. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"festival_content": call_gemini(prompt)}

def generate_local_case_study(topic, region, language, grade_level):
    prompt = (
        f"Create a case study in {language} about {topic} specific to {region} "
        f"for {grade_level} students. Include local challenges, solutions, and outcomes. "
        "Make it relatable with realistic scenarios. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"case_study": call_gemini(prompt)}

def call_gemini(prompt, model_name=None):
    response = model.generate_content(prompt)
    return response.text
class SahayakAgent:
    def __init__(self):
        self.name = "Sahayak"
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = (
            f"You are acting as {self.name}. You are an AI teaching assistant for multi-grade classrooms in rural India. "
            "Specialize in generating hyperlocal, culturally relevant content in Indian languages. "
            "Always consider: grade level appropriateness, cultural sensitivity, and regional context. "
            "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If the user specifies a target language, reply in that language. If the user gives a prompt in English, reply in English unless another language is specified. If the reply is in a non-English language, do not provide an English translation—just reply in the specified/generated language. Never translate your output to English unless explicitly asked. "
            "When generating content: "
            "- Use local names, landmarks, festivals, and traditions "
            "- Include moral values aligned with Indian education "
            "- Make content relatable to village/rural life "
            "- Prefer simple language with local idioms "
            "If you don't know an answer, use record_unknown_question."
        )
        return system_prompt

    def chat(self, message, history):
        prompt = self.system_prompt() + "\nUser: " + message

        response = model.generate_content(prompt)
        return response.text

def transcribe_audio(audio_filepath, language_code="en-US"):
    client = speech.SpeechClient()
    with open(audio_filepath, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript
    return transcript
import gradio as gr

def chat_with_voice_or_text(input_type, text_input, audio_input, language, history):
    if input_type == "Text":
        user_message = text_input
    else:
        if audio_input is None:
            return "Please record a message.", history
        user_message = transcribe_audio(audio_input, language)
    return agent.chat(user_message, history)
input_type = gr.Radio(["Text", "Audio"], value="Text", label="Input Type")
text_input = gr.Textbox(label="Type your message")
audio_input = gr.Audio(type="filepath", label="Record your message", visible=False)
language = gr.Dropdown(
    [
        "en-US", "hi-IN", "ta-IN", "te-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN"
    ],
    value="en-US",
    label="Speech Recognition Language",
    visible=False  
)

if __name__ == "__main__":
    agent = SahayakAgent()
    with gr.Blocks() as demo:
        gr.Markdown("# Sahayak Chat (Text & Audio)")
        input_type = gr.Radio(["Text", "Audio"], value="Text", label="Input Type")
        text_input = gr.Textbox(label="Type your message")
        audio_input = gr.Audio(type="filepath", label="Record your message", visible=False)
        language = gr.Dropdown(
            [
                "en-US", "hi-IN", "ta-IN", "te-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN"
            ],
            value="en-US",
            label="Speech Recognition Language",
            visible=False 
        )
        chatbot = gr.Chatbot()
        state = gr.State([])

        def toggle_language_dropdown(input_type):
            visible = (input_type == "Audio")
            return gr.update(visible=visible), gr.update(visible=visible)

        def respond(input_type, text_input, audio_input, language, history):
            if input_type == "Text":
                user_message = text_input
            else:
                if audio_input is None:
                    return history + [["Please record a message.", ""]], history
                user_message = transcribe_audio(audio_input, language)
            response = agent.chat(user_message, history)
            history = history + [[user_message, response]]
            return history, history

        input_type.change(
            toggle_language_dropdown,
            inputs=input_type,
            outputs=[language, audio_input]
        )

        submit_btn = gr.Button("Send")
        submit_btn.click(
            respond,
            [input_type, text_input, audio_input, language, state],
            [chatbot, state]
        )
    demo.launch()