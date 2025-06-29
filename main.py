import os
import asyncio
import tempfile
import requests
import io
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from mcp_use import MCPAgent, MCPClient
from QuestionGenerator import get_gemini_question_generator
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid
from typing import Dict, Any

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
      "url": "http://ao.uptopoint.net:7192/mcp/t-jaGTNm4VVMCLKHDF/sse",
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
    *   If a response is unclear, incomplete, or touches on a point needing deeper exploration relevant to the question, ask a polite follow-up question to probe further. Try to stick to the core intent of the question set, but you may use additional tools or resources (including AgentZero or MCP tools) if the user requests it or if it would help the interview process.
    *   If the response is sufficient, acknowledge it briefly (e.g., "Thank you," "Okay") and move on to the next question in the list.
    *   You may use external tools, AgentZero, or MCP tools if the user asks for it, or if it would improve the interview or provide a better experience. If the user asks you to use any of these tools, YOU MUST USE IT
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
agent = MCPAgent(llm=llm, client=client, max_steps=30, system_prompt=system_prompt, memory_enabled=True)

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

# In-memory store for subordinate agent tasks and results
subordinate_tasks: Dict[str, Dict[str, Any]] = {}

# In-memory store for tasks and results
query_tasks: Dict[str, Dict[str, Any]] = {}

@app.post("/query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a query as a background task. Always returns a task_id, status, and result (if ready).
    """
    task_id = str(uuid.uuid4())
    try:
        coro = agent.run(req.query)
        result = await asyncio.wait_for(asyncio.shield(coro), timeout=0.01)
        query_tasks[task_id] = {"status": "completed", "result": result}
        return {"task_id": task_id, "status": "completed", "result": result}
    except asyncio.TimeoutError:
        query_tasks[task_id] = {"status": "running", "result": None}
        async def run_task():
            try:
                result = await agent.run(req.query)
                query_tasks[task_id] = {"status": "completed", "result": result}
            except Exception as e:
                query_tasks[task_id] = {"status": "error", "result": str(e)}
        background_tasks.add_task(run_task)
        return {"task_id": task_id, "status": "running", "result": None}
    except Exception as e:
        query_tasks[task_id] = {"status": "error", "result": str(e)}
        return {"task_id": task_id, "status": "error", "result": str(e)}

@app.get("/query_result/{task_id}")
async def get_query_result(task_id: str):
    """
    Poll for the result of a query by task_id.
    """
    task = query_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using Groq API
    """
    try:
        logger.info("Processing audio transcription request")
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            # Write the uploaded audio to the temporary file
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Get Groq API key from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")

        # Set up the headers for the Groq API request
        headers = {
            "Authorization": f"Bearer {groq_api_key}"
        }

        # Prepare the form data for the Groq API request
        with open(temp_audio_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(temp_audio_path), audio_file, "audio/wav")
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
                data=data
            )
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Groq API error: {response.text}"
            )
        
        # Parse the response
        result = response.json()
        transcribed_text = result.get("text", "").strip()
        logger.info("Transcription completed successfully")
        
        return {"text": transcribed_text}
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # Disable reload to avoid multiple server instances
