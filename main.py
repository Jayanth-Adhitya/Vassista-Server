import os
import asyncio
import tempfile
import requests
import io
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from mcp_use import MCPAgent, MCPClient
from QuestionGenerator import get_gemini_question_generator
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def startup_generate_questions():
    try:
        user_context_path = os.path.join(os.path.dirname(__file__), 'user_context.json')
        with open(user_context_path, 'r', encoding='utf-8') as f:
            user_context = json.load(f)
        generator = get_gemini_question_generator()
        questions = await generator.generate_questions(
            interview_type=user_context.get("INTERVIEW_TYPE", ""),
            resume=user_context.get("RESUME", ""),
            cover_letter=user_context.get("COVER_LETTER", ""),
            background_story=user_context.get("BACKGROUND_STORY", ""),
            num_questions=user_context.get("NUM_QUESTIONS", 6)
        )
        context_path = os.path.join(os.path.dirname(__file__), 'interview_context.json')
        if os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                interview_context = json.load(f)
        else:
            interview_context = {}
        interview_context["QUESTION_LIST"] = questions
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(interview_context, f, indent=2)
        logger.info("Questions generated and interview_context.json updated on startup.")
    except Exception as e:
        logger.error(f"Startup question generation failed: {str(e)}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    asyncio.create_task(startup_generate_questions())
    yield

# Initialize FastAPI
app = FastAPI(
    title="MCP Proxy Server",
    description="Proxy API to query LLM + MCP agents with voice capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    # Allow requests from the client (running on localhost:3000)
    allow_origins=["*"],
    # Allow all types of request methods (GET, POST, etc.)
    allow_methods=["*"],
    # Allow standard headers
    allow_headers=["*"],
)

# Configuration for MCP servers 
CLIENT_CONFIG = {
  "mcpServers": {
    "clickup": {
      "type": "stdio",  # same as local
      "command": "npx",
      "args": ["-y", "clickup-mcp-server"],
      "env": {"CLICKUP_API_TOKEN": os.getenv("CLICKUP_API_TOKEN")}
    },
    "agent-zero": {
      "type": "sse",
      "url": "https://ao.uptopoint.net/mcp/t-0/sse",
      # optionally:
      # "authorization_token": os.getenv("AGENT_ZERO_TOKEN")
    }
  }
}

# Read interview context from a JSON file
CONTEXT_JSON_PATH = os.path.join(os.path.dirname(__file__), 'interview_context.json')
if os.path.exists(CONTEXT_JSON_PATH):
    with open(CONTEXT_JSON_PATH, 'r', encoding='utf-8') as f:
        INTERVIEW_CONTEXT = json.load(f)
else:
    INTERVIEW_CONTEXT = {}

# Build the system prompt by substituting variables from INTERVIEW_CONTEXT
System_Prompt_Template = """# SYSTEM ROLE: AI Interviewer

## 1. Persona:
You are a professional and objective interviewer representing the hiring organization. Your tone should be friendly, encouraging, and strictly professional. You are here to conduct a structured interview for a specific position (details provided by the user).

## 2. Objective:
Your primary goal is to assess the candidate's suitability for the role they are interviewing for by asking pre-defined questions and evaluating their responses based on the provided criteria. You must facilitate a focused conversation that allows the candidate to showcase their relevant skills, experience, and behavioral traits.

## 3. Context:
- Role: {POSITION_DESCRIPTION}
- Hiring Organization: (Details will be provided by the user, or kept generic)
- Candidate Name: {CANDIDATE_NAME}
- Interview Structure: You will follow the steps outlined below.
- Question Set: You have been provided with a specific list of questions to ask.

## 4. Interview Process & Rules:

1.  **Introduction:** Start the interview by welcoming the candidate, introducing yourself (you may use a generic or provided name), mentioning the purpose of the interview (for the position they are interviewing for), and briefly explaining the interview structure and timeframe (if applicable).
2.  **Questioning Phase:**
    *   Ask questions one at a time from the provided {QUESTION_LIST}.
    *   Wait for the candidate's complete response before proceeding.
    *   Listen carefully and analyze the response based on {EVALUATION_FOCUS}.
    *   If a response is unclear, incomplete, or touches on a point needing deeper exploration relevant to the question, ask a polite follow-up question to probe further. *Do not deviate significantly from the core intent of the question set.*
    *   If the response is sufficient, acknowledge it briefly (e.g., "Thank you," "Okay") and move on to the next question in the list.
    *   *Strict Rule:* You must stick to the provided {QUESTION_LIST} as much as possible. Only generate follow-ups to clarify or elaborate on the candidate's answer to *those specific questions*. Do not introduce entirely new topics or questions outside the list unless absolutely necessary for basic clarification.
3.  **Candidate Questions (Optional/At End):** Unless specifically instructed otherwise by the user input, defer candidate questions until the end of the main questioning phase. If the candidate asks a question mid-interview, politely note it and say you will address questions at the end. Assume you are NOT equipped to answer complex candidate questions about the role or company. Politely state that you will collect their questions to pass along to the hiring manager or relevant person.
4.  **Closing:** Once all questions from {QUESTION_LIST} have been asked and follow-ups explored, thank the candidate for their time and participation. Briefly explain that the hiring team will review their responses and communicate next steps. End the conversation politely.

## 5. Evaluation Focus (Implicit During Conversation):
As you listen to responses, consider how well they demonstrate the traits, skills, and experiences implied by the questions in {QUESTION_LIST}. Look for:
- Relevance of experience and skills.
- Clarity and structure of communication.
- Specific examples supporting claims (e.g., using methods like STAR).
- Problem-solving approach (if applicable to questions).
- Understanding of relevant concepts (if applicable to questions).
- Professionalism in communication.
- How well the answers directly address the question asked.

*Note: You are not required to output an explicit evaluation score or summary.* Your task is to *conduct* the interview effectively to gather the necessary information.

## 6. Constraints & Etiquette:
- Speak *only* as the interviewer. Do not reveal you are an AI. You may use a simple name like "Interviewer" or one provided by the user.
- Ask one question (or follow-up) per turn. Wait for the candidate's complete response.
- Maintain a positive, neutral, and professional tone throughout.
- Avoid revealing any bias or pre-judgment.
- Do not provide feedback or opinions on the candidate's answers during the interview ("That's a great answer," "You did well"). Acknowledge briefly and move on or follow up.
- Manage the flow based on the candidate's responses and the question list.
- If the candidate goes significantly off-topic relative to the question, gently steer them back.
- If the candidate indicates they don't know the answer or cannot provide an example, acknowledge it and move on.

## 7. Starting the Interview:
Your first output will be the introduction as described in step 4.1.

## 8. Placeholders to be Provided by User Input:
- INTERVIEWER_NAME: {INTERVIEWER_NAME}
- POSITION_DESCRIPTION: {POSITION_DESCRIPTION}
- CANDIDATE_NAME: {CANDIDATE_NAME}
- ESTIMATED_DURATION: {ESTIMATED_DURATION}
- QUESTION_LIST: {QUESTION_LIST}
- EVALUATION_FOCUS: {EVALUATION_FOCUS}
"""

# Format QUESTION_LIST for prompt
question_list = INTERVIEW_CONTEXT.get("QUESTION_LIST", [])
if isinstance(question_list, list):
    formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(question_list))
else:
    formatted_questions = str(question_list)

# Build the system prompt with context
system_prompt = System_Prompt_Template.format(
    INTERVIEWER_NAME=INTERVIEW_CONTEXT.get("INTERVIEWER_NAME", "Interviewer"),
    POSITION_DESCRIPTION=INTERVIEW_CONTEXT.get("POSITION_DESCRIPTION", ""),
    CANDIDATE_NAME=INTERVIEW_CONTEXT.get("CANDIDATE_NAME", ""),
    ESTIMATED_DURATION=INTERVIEW_CONTEXT.get("ESTIMATED_DURATION", ""),
    QUESTION_LIST=formatted_questions,
    EVALUATION_FOCUS=INTERVIEW_CONTEXT.get("EVALUATION_FOCUS", "")
)

client = MCPClient.from_dict(CLIENT_CONFIG)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
agent = MCPAgent(llm=llm, client=client, max_steps=30, system_prompt=system_prompt, memory_enabled=True, verbose=True)

async def check_servers():
    servers = await client.list_servers()
    print("Connected MCP servers:", servers)


# Request models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "Fritz-PlayAI"  # Default voice

class TranscriptionResponse(BaseModel):
    text: str

@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """
    Handle incoming query, run through MCPAgent, and return result.
    """
    try:
        logger.info(f"Received query: {req.query}")
        # Use the same approach as your working code
        result = await agent.run(req.query)
        logger.info(f"Query processed successfully")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using Groq API - FIXED VERSION
    """
    try:
        logger.info(f"Processing audio transcription request - File: {audio.filename}, Content-Type: {audio.content_type}")
        
        # Validate the uploaded file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read the file content
        audio_content = await audio.read()
        logger.info(f"Audio file size: {len(audio_content)} bytes")
        
        if len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name
        
        logger.info(f"Saved audio to temporary file: {temp_audio_path}")

        # Get Groq API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            # Clean up temp file
            os.unlink(temp_audio_path)
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")

        # Set up the headers for the Groq API request
        headers = {
            "Authorization": f"Bearer {groq_api_key}"
        }

        try:
            # Prepare the form data for the Groq API request
            with open(temp_audio_path, "rb") as audio_file:
                files = {
                    "file": (audio.filename or "recording.wav", audio_file, "audio/wav")
                }
                
                data = {
                    "model": "whisper-large-v3-turbo",
                    "response_format": "json",
                    "language": "en",
                    "temperature": 0.0
                }
                
                # Make the request to Groq API
                logger.info("Sending request to Groq API")
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30  # Add timeout
                )
            
            # Clean up the temporary file
            os.unlink(temp_audio_path)
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.error(f"Groq API error: Status {response.status_code}, Response: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Groq API error: {response.text}"
                )
            
            # Parse the response
            result = response.json()
            transcribed_text = result.get("text", "").strip()
            logger.info(f"Transcription completed successfully: '{transcribed_text[:100]}...'")
            
            return {"text": transcribed_text}
        
        except requests.exceptions.RequestException as e:
            # Clean up temp file in case of request error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            logger.error(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/speak")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using Groq's TTS API
    """
    try:
        logger.info("Processing text-to-speech request using Groq TTS")
        
        # Get Groq API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")
        
        # Set up the headers for the Groq API request
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        data = {
            "model": "playai-tts",
            "voice": request.voice,
            "input": request.text,
            "response_format": "wav"
        }
        
        # Make the request to Groq TTS API
        logger.info(f"Sending TTS request to Groq API with voice: {request.voice}")
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers=headers,
            json=data
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Groq TTS API error: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq TTS API error: {response.text}"
            )
        
        # Get the audio content
        audio_content = io.BytesIO(response.content)
        audio_content.seek(0)

        logger.info("Text-to-speech conversion completed successfully with Groq TTS")
        
        # Return the audio as a streaming response
        return StreamingResponse(
            audio_content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        logger.error(f"Groq TTS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Groq TTS error: {str(e)}")

@app.post("/generate_questions")
async def generate_questions():
    """
    Generate interview questions using the QuestionGenerator and update the interview context.
    """
    try:
        # Load user context
        user_context_path = os.path.join(os.path.dirname(__file__), 'user_context.json')
        with open(user_context_path, 'r', encoding='utf-8') as f:
            user_context = json.load(f)

        generator = get_gemini_question_generator()
        questions = await generator.generate_questions(
            interview_type=user_context.get("INTERVIEW_TYPE", ""),
            resume=user_context.get("RESUME", ""),
            cover_letter=user_context.get("COVER_LETTER", ""),
            background_story=user_context.get("BACKGROUND_STORY", ""),
            num_questions=user_context.get("NUM_QUESTIONS", 6)
        )

        # Update interview_context.json
        context_path = os.path.join(os.path.dirname(__file__), 'interview_context.json')
        if os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                interview_context = json.load(f)
        else:
            interview_context = {}
        interview_context["QUESTION_LIST"] = questions
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(interview_context, f, indent=2)
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Proxy Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False) 
    # Disable reload to avoid multiple server instances
    asyncio.run(check_servers())
