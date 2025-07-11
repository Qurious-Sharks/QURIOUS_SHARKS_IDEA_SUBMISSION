from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from google.cloud import speech


load_dotenv(override=True)

def transcribe_audio(audio_filepath, language_code="en-US"):
    """Transcribe audio file to text using Google Cloud Speech API"""
    try:
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
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def call_gemini(prompt):
    """Call Gemini API with the given prompt and return the response"""
    try:
        gemini = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"), 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = gemini.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating content: {str(e)}"


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
        "required": ["topic", "language", "grade_level"],
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
        "required": ["language", "topic"],
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
        "required": ["topic", "language"],
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
def generate_local_story(topic, language, grade_level, tone="inspiring", region="India"):
    prompt = (
        f"Create a {tone} story in {language} for {grade_level} students about {topic}. "
        f"Set it in {region} with local landmarks, cultural elements, and traditional values. "
        "Include characters with local names and incorporate regional festivals or traditions. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"story": call_gemini(prompt)}

def generate_local_poem(topic, language, grade_level, poem_type="rhyming"):
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

def generate_local_riddles(language, topic, difficulty="medium"):
    prompt = (
        f"Create 3 {difficulty} riddles in {language} about {topic}. "
        "Include answers. Make them culturally relevant with local references. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"riddles": call_gemini(prompt)}

def generate_local_song(topic, language, melody_type="folk"):
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




def generate_local_quiz(topic, language, grade_level, question_type="MCQ", num_questions=5):
    prompt = (
        f"Create {num_questions} {question_type} questions in {language} about {topic} "
        f"for {grade_level} students. Include answers. Use local examples and contexts. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"quiz": call_gemini(prompt)}

def generate_local_game(topic, language, materials="paper and markers", duration=15):
    prompt = (
        f"Create a {duration}-minute educational game in {language} about {topic}. "
        f"Materials available: {materials}. Include rules, setup, and learning objectives. "
        "Use traditional game formats with local cultural elements. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"game": call_gemini(prompt)}

def generate_local_festival_content(festival, language, grade_level="5th grade"):
    prompt = (
        f"Create educational content in {language} about {festival} for {grade_level} students. "
        "Include its significance, traditions, related activities, and a simple craft idea. "
        "Connect to local practices and values. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"festival_content": call_gemini(prompt)}

def generate_local_case_study(topic, region, language, grade_level="5th grade"):
    prompt = (
        f"Create a case study in {language} about {topic} specific to {region} "
        f"for {grade_level} students. Include local challenges, solutions, and outcomes. "
        "Make it relatable with realistic scenarios. "
        "LANGUAGE POLICY: If the user prompt is in a language other than English and does not specify a target language, reply in the same language as the prompt. If a language is specified, reply only in that language. Do not provide English translations unless explicitly asked."
    )
    return {"case_study": call_gemini(prompt)}

def record_unknown_question(question):
    """Record a question the AI couldn't answer"""
    # For now, just print the question. In a real implementation, you might want to log it to a file or database
    print(f"Unknown question recorded: {question}")
    return {"status": "recorded", "question": question}

class Me:

    def __init__(self):
        self.name = "Sahayak"
        reader = PdfReader("/Users/hemasurya/Desktop/Profile_chat/Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("/Users/hemasurya/Desktop/Profile_chat/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            
            if tool:
                # Get function signature to check required parameters
                import inspect
                sig = inspect.signature(tool)
                required_params = [name for name, param in sig.parameters.items() 
                                 if param.default == inspect.Parameter.empty]
                
                # Check if all required parameters are provided
                missing_params = [param for param in required_params if param not in arguments]
                
                if missing_params:
                    # Return a special response indicating missing parameters
                    missing_info = {
                        "missing_parameters": missing_params,
                        "tool_name": tool_name,
                        "provided_arguments": arguments,
                        "status": "needs_parameters"
                    }
                    results.append({"role": "tool", "content": json.dumps(missing_info), "tool_call_id": tool_call.id})
                else:
                    # All required parameters are present, execute the tool
                    try:
                        result = tool(**arguments)
                        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
                    except Exception as e:
                        error_result = {"error": str(e), "status": "execution_failed"}
                        results.append({"role": "tool", "content": json.dumps(error_result), "tool_call_id": tool_call.id})
            else:
                error_result = {"error": f"Tool {tool_name} not found", "status": "tool_not_found"}
                results.append({"role": "tool", "content": json.dumps(error_result), "tool_call_id": tool_call.id})
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
            "PARAMETER HANDLING: When calling tools, if required parameters are missing, the system will automatically ask the user for them. You should call tools with whatever parameters you can extract from the user's request. If the user says 'use defaults' or provides incomplete information, the system will handle this appropriately. "
            "If you don't know an answer, use record_unknown_question."
        )
        return system_prompt
    
    def chat(self, message, history):
        # Convert Gradio history format to OpenAI format
        messages = [{"role": "system", "content": self.system_prompt()}]
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        done = False
        gemini = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"), 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        while not done:
            response = gemini.chat.completions.create(model="gemini-2.0-flash", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
                
                # Check if any tool calls need parameters
                needs_parameters = False
                for result in results:
                    try:
                        result_content = json.loads(result["content"])
                        if result_content.get("status") == "needs_parameters":
                            needs_parameters = True
                            break
                    except:
                        continue
                
                if needs_parameters:
                    # Ask user for missing parameters
                    return self.ask_for_missing_parameters(results, history)
            else:
                done = True
        return response.choices[0].message.content
    
    def ask_for_missing_parameters(self, results, history):
        """Ask user for missing parameters and provide default options"""
        missing_info = None
        for result in results:
            try:
                result_content = json.loads(result["content"])
                if result_content.get("status") == "needs_parameters":
                    missing_info = result_content
                    break
            except:
                continue
        
        if not missing_info:
            return "I encountered an error processing your request. Please try again."
        
        tool_name = missing_info["tool_name"]
        missing_params = missing_info["missing_parameters"]
        provided_args = missing_info["provided_arguments"]
        
        # Create a helpful message asking for missing parameters
        message = f"I need some additional information to generate your {tool_name.replace('_', ' ')}. "
        message += f"Please provide the following missing details:\n\n"
        
        for param in missing_params:
            # Provide helpful context for each parameter
            if param == "tone":
                message += f"• **{param}**: Choose from funny, serious, inspiring, or adventurous\n"
            elif param == "poem_type":
                message += f"• **{param}**: Choose from rhyming, haiku, free verse, or limerick\n"
            elif param == "question_type":
                message += f"• **{param}**: Choose from MCQ, true/false, or short answer\n"
            elif param == "difficulty":
                message += f"• **{param}**: Choose from easy, medium, or hard\n"
            elif param == "melody_type":
                message += f"• **{param}**: Choose from folk, classical, modern, or traditional\n"
            elif param == "num_questions":
                message += f"• **{param}**: Number of questions (1-10)\n"
            elif param == "duration":
                message += f"• **{param}**: Duration in minutes (5-60)\n"
            elif param == "grade_level":
                message += f"• **{param}**: Grade level (e.g., 1st grade, 5th grade, high school)\n"
            elif param == "region":
                message += f"• **{param}**: Region or state (e.g., Maharashtra, Tamil Nadu, Karnataka)\n"
            elif param == "characters":
                message += f"• **{param}**: Character names or descriptions\n"
            elif param == "context":
                message += f"• **{param}**: Context or setting for the dialogue\n"
            elif param == "scenario":
                message += f"• **{param}**: Scenario description\n"
            elif param == "materials":
                message += f"• **{param}**: Available materials (e.g., paper, markers, cards)\n"
            elif param == "festival":
                message += f"• **{param}**: Festival name (e.g., Diwali, Holi, Pongal)\n"
            else:
                message += f"• **{param}**: Please specify\n"
        
        message += f"\n**Current information provided:**\n"
        for key, value in provided_args.items():
            message += f"• {key}: {value}\n"
        
        message += f"\nYou can either:\n"
        message += f"1. Provide the missing details\n"
        message += f"2. Say 'use defaults' to generate with reasonable defaults\n"
        message += f"3. Say 'cancel' to stop this request"
        
        return message
    
    def handle_parameter_response(self, user_response, history):
        """Handle user response to parameter request"""
        user_response_lower = user_response.lower().strip()
        
        # Check for special commands
        if "cancel" in user_response_lower:
            return "Request cancelled. How else can I help you?"
        
        if "use defaults" in user_response_lower or "defaults" in user_response_lower:
            return self.generate_with_defaults(history)
        
        # Extract parameters from user response
        # This is a simplified parser - in a real implementation, you might want more sophisticated parsing
        extracted_params = self.extract_parameters_from_response(user_response)
        
        if extracted_params:
            return self.generate_with_extracted_params(extracted_params, history)
        else:
            return "I couldn't understand the parameters you provided. Please try again with clearer information, or say 'use defaults' to generate with reasonable defaults."
    
    def extract_parameters_from_response(self, user_response):
        """Extract parameters from user response"""
        params = {}
        user_response_lower = user_response.lower()
        
        # Extract tone
        if "funny" in user_response_lower:
            params["tone"] = "funny"
        elif "serious" in user_response_lower:
            params["tone"] = "serious"
        elif "inspiring" in user_response_lower:
            params["tone"] = "inspiring"
        elif "adventurous" in user_response_lower:
            params["tone"] = "adventurous"
        
        # Extract poem type
        if "rhyming" in user_response_lower:
            params["poem_type"] = "rhyming"
        elif "haiku" in user_response_lower:
            params["poem_type"] = "haiku"
        elif "free verse" in user_response_lower:
            params["poem_type"] = "free verse"
        elif "limerick" in user_response_lower:
            params["poem_type"] = "limerick"
        
        # Extract question type
        if "mcq" in user_response_lower or "multiple choice" in user_response_lower:
            params["question_type"] = "MCQ"
        elif "true/false" in user_response_lower or "true false" in user_response_lower:
            params["question_type"] = "true/false"
        elif "short answer" in user_response_lower:
            params["question_type"] = "short answer"
        
        # Extract difficulty
        if "easy" in user_response_lower:
            params["difficulty"] = "easy"
        elif "medium" in user_response_lower:
            params["difficulty"] = "medium"
        elif "hard" in user_response_lower:
            params["difficulty"] = "hard"
        
        # Extract melody type
        if "folk" in user_response_lower:
            params["melody_type"] = "folk"
        elif "classical" in user_response_lower:
            params["melody_type"] = "classical"
        elif "modern" in user_response_lower:
            params["melody_type"] = "modern"
        elif "traditional" in user_response_lower:
            params["melody_type"] = "traditional"
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+', user_response)
        if numbers:
            if "question" in user_response_lower or "questions" in user_response_lower:
                params["num_questions"] = min(10, max(1, int(numbers[0])))
            if "minute" in user_response_lower or "duration" in user_response_lower:
                params["duration"] = min(60, max(5, int(numbers[0])))
        
        # Extract grade level
        grade_patterns = [
            r'(\d+)(?:st|nd|rd|th)\s*grade',
            r'grade\s*(\d+)',
            r'class\s*(\d+)',
            r'(\d+)\s*class'
        ]
        for pattern in grade_patterns:
            match = re.search(pattern, user_response_lower)
            if match:
                grade_num = int(match.group(1))
                if grade_num <= 12:
                    params["grade_level"] = f"{grade_num}th grade"
                break
        
        # Extract region/state names (common Indian states)
        states = [
            "maharashtra", "tamil nadu", "karnataka", "kerala", "andhra pradesh", 
            "telangana", "gujarat", "rajasthan", "madhya pradesh", "uttar pradesh",
            "bihar", "west bengal", "odisha", "assam", "punjab", "haryana",
            "himachal pradesh", "uttarakhand", "jharkhand", "chhattisgarh"
        ]
        for state in states:
            if state in user_response_lower:
                params["region"] = state.title()
                break
        
        # Extract festival names
        festivals = [
            "diwali", "holi", "pongal", "onam", "baisakhi", "rakhi", "ganesh chaturthi",
            "durga puja", "ram navami", "krishna janmashtami", "makar sankranti"
        ]
        for festival in festivals:
            if festival in user_response_lower:
                params["festival"] = festival.title()
                break
        
        return params
    
    def generate_with_defaults(self, history):
        """Generate content with default parameters"""
        # Find the original request in history
        original_request = None
        for i in range(len(history) - 1, -1, -1):
            if "I need some additional information" not in history[i][1]:
                original_request = history[i][0]
                break
        
        if not original_request:
            return "I couldn't find your original request. Please try again."
        
        # Add default parameters to the original request
        enhanced_request = original_request + " (use reasonable defaults for any missing parameters)"
        return self.chat(enhanced_request, history[:-2])  # Remove the parameter request and response
    
    def generate_with_extracted_params(self, extracted_params, history):
        """Generate content with extracted parameters"""
        # Find the original request in history
        original_request = None
        for i in range(len(history) - 1, -1, -1):
            if "I need some additional information" not in history[i][1]:
                original_request = history[i][0]
                break
        
        if not original_request:
            return "I couldn't find your original request. Please try again."
        
        # Add extracted parameters to the original request
        param_strings = []
        for key, value in extracted_params.items():
            param_strings.append(f"{key}: {value}")
        
        enhanced_request = original_request + f" ({', '.join(param_strings)})"
        return self.chat(enhanced_request, history[:-2])  # Remove the parameter request and response
    

if __name__ == "__main__":
    me = Me()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Sahayak- AI Teaching Assistant (Text & Audio)")
        
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
            
            # Check if this is a response to a parameter request
            if history and "I need some additional information" in history[-1][1]:
                # This is a response to parameter request
                response = me.handle_parameter_response(user_message, history)
            else:
                # Normal chat
                response = me.chat(user_message, history)
            
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
    
