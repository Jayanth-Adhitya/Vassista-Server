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
from fastapi.middleware.cors import CORSMiddleware
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
# AgentZero direct connection configuration
AGENT_ZERO_URL = "https://ao.uptopoint.net"
# Request models
class QueryRequest(BaseModel):
    context: str
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
    
    # Log the received request for debugging
    logger.info(f"Received query request: {req.context}")

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
            # Log the constructed query for debugging
            logger.info(f"Constructed query for AgentZero: {req.context}")
            
            payload = {"text": req.context}
            
            agent_response = requests.post(
                message_url, 
                headers=headers, 
                cookies=cookies, 
                json=payload, 
                verify=False # verify=False for self-signed certs if any
            )
            
            # Log the full response from AgentZero for debugging
            logger.info(f"AgentZero response status: {agent_response.status_code}")
            logger.info(f"AgentZero response content: {agent_response.text}")
            
            agent_response.raise_for_status() # This will raise an exception for bad status codes
            
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
        logger.info("Processing text-to-speech request using AgentZero's /synthesize endpoint")
        
        # Prepare the request payload for AgentZero's /synthesize endpoint
        # AgentZero's synthesize.py expects 'text' and optionally 'ctxid'
        payload = {
            "text": request.text,
            # "ctxid": "some_context_id_if_needed" # Add ctxid if your AgentZero setup uses it
        }
        
        # Send request to AgentZero's /synthesize endpoint
        synthesize_url = f"{AGENT_ZERO_URL}/synthesize"
        
        # Fetch CSRF token and cookies for AgentZero
        csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
        csrf_response = requests.get(csrf_url, verify=False)
        csrf_response.raise_for_status()
        
        csrf_json = csrf_response.json()
        csrf_token = csrf_json.get("token")
        cookies = csrf_response.cookies
        
        if not csrf_token:
            raise ValueError(f"CSRF token not found for TTS: {csrf_json}")

        headers = {
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json"
        }

        logger.info(f"Sending TTS request to AgentZero at {synthesize_url}")
        response = requests.post(
            synthesize_url,
            headers=headers,
            cookies=cookies,
            json=payload,
            verify=False
        )
        
        response.raise_for_status() # Raise an exception for bad status codes
        
        agent_tts_response = response.json()
        
        if not agent_tts_response.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"AgentZero TTS error: {agent_tts_response.get('error', 'Unknown error')}"
            )
        
        # AgentZero returns base64 encoded audio
        base64_audio = agent_tts_response.get("audio")
        if not base64_audio:
            raise HTTPException(status_code=500, detail="No audio content received from AgentZero TTS")
        
        audio_content = io.BytesIO(base64.b64decode(base64_audio))
        audio_content.seek(0)
        logger.info("Text-to-speech conversion completed successfully with AgentZero TTS")
        
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
