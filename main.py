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
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid
from typing import Dict, Any
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
    allow_headers=["*"],
)
# Configuration for MCP servers 
CLIENT_CONFIG = {
    "mcpServers": {
        "agent-zero": {
            "type": "sse",
            "url": "http://aotest.uptopoint.net:7777/mcp/t-jaGTNm4VVMCLKHDF/sse",
            # "authorization_token": os.getenv("AGENT_ZERO_TOKEN")
        }
    }
}
# Set a clear system prompt for AgentZero Assistant
system_prompt = """
You are a proxy for the AgentZero tool. Your only function is to receive a query and pass it to the AgentZero tool. You must always use the AgentZero tool. Do not answer directly.
"""
# Initialize ChatGoogleGenerativeAI as the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17",api_key=os.getenv("GOOGLE_API_KEY"))
client = MCPClient.from_dict(CLIENT_CONFIG)
# Only expose the AgentZero tool and instruct the agent to always use it
agent = MCPAgent(
    llm=llm,
    client=client,
    max_steps=30,
    system_prompt=system_prompt,
    memory_enabled=True
)
# AgentZero direct connection configuration
AGENT_ZERO_URL = "https://aotest.uptopoint.net"
# Request models
class Message(BaseModel):
    id: str
    sender: str
    text: str

class Notification(BaseModel):
    id: str
    title: str
    body: str
    timestamp: Any

class QueryRequest(BaseModel):
    query: str
    messages: list[Message] = []
    notifications: list[Notification] = []
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
    query_tasks[task_id] = {"status": "running", "result": None}
    async def run_task():
        try:
            # Fetch CSRF token and cookies
            csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
            csrf_response = requests.get(csrf_url, verify=False) # verify=False for self-signed certs if any
            csrf_response.raise_for_status()
            
            csrf_json = csrf_response.json()
            
            csrf_token = csrf_json.get("token") # Corrected key from "csrf_token" to "token"
            cookies = csrf_response.cookies
            
            if not csrf_token:
                raise ValueError(f"CSRF token not found in response. Full response: {csrf_json}")

            # Send message to AgentZero
            message_url = f"{AGENT_ZERO_URL}/message"
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }
            # Construct the context string
            context_parts = [f"User Query: {req.query}"]
            
            if req.messages:
                context_parts.append("\nRecent Messages:")
                for msg in req.messages:
                    context_parts.append(f"- {msg.sender}: {msg.text}")
            
            if req.notifications:
                context_parts.append("\nNotifications:")
                for notif in req.notifications:
                    context_parts.append(f"- {notif.title}: {notif.body}")
            
            full_query = "\n".join(context_parts)
            
            payload = {"text": full_query}
            
            agent_response = requests.post(
                message_url, 
                headers=headers, 
                cookies=cookies, 
                json=payload, 
                verify=False # verify=False for self-signed certs if any
            )
            agent_response.raise_for_status()
            
            agent_result_json = agent_response.json()
            # Extract only the 'message' content from the AgentZero response
            result_text = agent_result_json.get("message", str(agent_result_json)) 
            query_tasks[task_id] = {"status": "completed", "result": result_text}
        except Exception as e:
            query_tasks[task_id] = {"status": "error", "result": str(e)}
    background_tasks.add_task(run_task)
    return {"task_id": task_id, "status": "running", "result": None}
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
if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Proxy Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # Disable reload to avoid multiple server instances
