import os
import asyncio
import tempfile
import requests
import io
import logging
import json
import base64
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List
import uuid

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import FastRTC voice agent after logging setup
try:
    from fastrtc_voice_agent import (
        initialize_agent, 
        get_fastrtc_stream, 
        update_mobile_context,
        mobile_context,
        voice_agent
    )
    FASTRTC_AVAILABLE = True
    logger.info("FastRTC voice agent imported successfully")
except Exception as e:
    FASTRTC_AVAILABLE = False
    logger.warning(f"FastRTC not available, using fallback: {e}")
    
    # Import fallback implementation
    from fastrtc_voice_agent_fallback import (
        initialize_agent, 
        get_fastrtc_stream, 
        update_mobile_context,
        mobile_context,
        voice_agent
    )
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize FastRTC voice agent on startup
    if FASTRTC_AVAILABLE:
        try:
            success = await initialize_agent()
            if success:
                logger.info("FastRTC voice agent initialized successfully")
            else:
                logger.error("Failed to initialize FastRTC voice agent")
        except Exception as e:
            logger.error(f"FastRTC initialization error: {e}")
    
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
# ================ FastRTC Configuration ================
# Enable local FastRTC streaming with Faster Whisper + Kokoro TTS
USE_FASTRTC_STREAMING = True  # Use FastRTC streaming system

# AgentZero external services (for AI responses, not TTS)
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
# Context cache for optimization
context_cache: Dict[str, Dict[str, Any]] = {}
cache_ttl = 300  # 5 minutes cache TTL

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_sessions: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections.append(websocket)
        self.connection_sessions[websocket] = session_id
        return session_id
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            del self.connection_sessions[websocket]
    
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            self.disconnect(websocket)
    
    async def send_stream_chunk(self, websocket: WebSocket, chunk_type: str, data: str, is_final: bool = False):
        message = {
            "type": chunk_type,
            "data": data,
            "is_final": is_final,
            "timestamp": time.time()
        }
        await self.send_message(websocket, message)

manager = ConnectionManager()

# Context optimization helpers
def get_cache_key(context: str) -> str:
    """Generate a cache key from context"""
    import hashlib
    return hashlib.md5(context.encode()).hexdigest()

def is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - cache_entry.get('timestamp', 0) < cache_ttl

def get_cached_response(context: str) -> str:
    """Get cached response if available and valid"""
    cache_key = get_cache_key(context)
    if cache_key in context_cache:
        entry = context_cache[cache_key]
        if is_cache_valid(entry):
            logger.info(f"Cache hit for key: {cache_key[:8]}...")
            return entry['response']
        else:
            # Remove expired entry
            del context_cache[cache_key]
    return None

def cache_response(context: str, response: str):
    """Cache a response"""
    cache_key = get_cache_key(context)
    context_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    logger.info(f"Cached response for key: {cache_key[:8]}...")

def clean_expired_cache():
    """Remove expired cache entries"""
    expired_keys = []
    for key, entry in context_cache.items():
        if not is_cache_valid(entry):
            expired_keys.append(key)
    
    for key in expired_keys:
        del context_cache[key]
    
    if expired_keys:
        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

def optimize_context(context: str) -> str:
    """Optimize context by removing redundant information and summarizing if too long"""
    # Clean up extra whitespace
    context = ' '.join(context.split())
    
    # If context is very long, we might want to summarize it
    max_length = 4000  # Reasonable limit
    if len(context) > max_length:
        logger.info(f"Context too long ({len(context)} chars), truncating...")
        # Keep the user query and most recent parts
        parts = context.split('---')
        if len(parts) > 1:
            # Keep user query and most recent sections
            user_query = parts[0] if 'User Query:' in parts[0] else ''
            recent_parts = parts[-2:] if len(parts) > 2 else parts[1:]
            context = user_query + '\n' + '\n---\n'.join(recent_parts)
        
        # Final truncation if still too long
        if len(context) > max_length:
            context = context[:max_length] + "... (truncated)"
    
    return context

# Streaming query function for WebSocket
async def stream_agent_response(websocket: WebSocket, context: str):
    try:
        # Clean expired cache entries periodically
        clean_expired_cache()
        
        # Optimize context
        optimized_context = optimize_context(context)
        await manager.send_stream_chunk(websocket, "status", "Optimizing context...")
        
        # Check cache first
        cached_response = get_cached_response(optimized_context)
        if cached_response:
            await manager.send_stream_chunk(websocket, "status", "Using cached response...")
            
            # Stream the cached response
            words = cached_response.split()
            chunk_size = 8  # Faster streaming for cached responses
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                is_final = i + chunk_size >= len(words)
                await manager.send_stream_chunk(websocket, "text", chunk, is_final)
                await asyncio.sleep(0.05)  # Faster for cached responses
                
            return cached_response
        
        await manager.send_stream_chunk(websocket, "status", "Fetching CSRF token...")
        
        # Fetch CSRF token and cookies
        csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
        csrf_response = requests.get(csrf_url, verify=False)
        csrf_response.raise_for_status()
        
        csrf_json = csrf_response.json()
        csrf_token = csrf_json.get("token")
        cookies = csrf_response.cookies
        
        if not csrf_token:
            raise ValueError(f"CSRF token not found in response. Full response: {csrf_json}")

        await manager.send_stream_chunk(websocket, "status", "Sending query to AgentZero...")
        
        # Send message to AgentZero
        message_url = f"{AGENT_ZERO_URL}/message"
        headers = {
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json"
        }
        
        payload = {"text": optimized_context}
        
        agent_response = requests.post(
            message_url, 
            headers=headers, 
            cookies=cookies, 
            json=payload, 
            verify=False
        )
        
        agent_response.raise_for_status()
        agent_result_json = agent_response.json()
        result_text = agent_result_json.get("message", str(agent_result_json))
        
        # Cache the response
        cache_response(optimized_context, result_text)
        
        # Stream the response in chunks for better UX
        words = result_text.split()
        chunk_size = 5  # Send 5 words at a time
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            is_final = i + chunk_size >= len(words)
            await manager.send_stream_chunk(websocket, "text", chunk, is_final)
            await asyncio.sleep(0.1)  # Small delay for streaming effect
            
        return result_text
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Streaming error: {error_msg}")
        await manager.send_stream_chunk(websocket, "error", error_msg, True)
        return error_msg

# Streaming TTS function
async def stream_tts_response(websocket: WebSocket, text: str):
    try:
        await manager.send_stream_chunk(websocket, "tts_status", "Preparing TTS...")
        
        # Fetch CSRF token for AgentZero
        csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
        csrf_response = requests.get(csrf_url, verify=False)
        csrf_response.raise_for_status()
        
        csrf_json = csrf_response.json()
        csrf_token = csrf_json.get("token")
        cookies = csrf_response.cookies
        
        if not csrf_token:
            raise ValueError(f"CSRF token not found for TTS: {csrf_json}")

        await manager.send_stream_chunk(websocket, "tts_status", "Generating audio...")
        
        # Split text into sentences for chunk-based TTS
        sentences = [s.strip() + "." for s in text.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 3:  # Skip very short sentences
                continue
                
            payload = {"text": sentence}
            synthesize_url = f"{AGENT_ZERO_URL}/synthesize"
            
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                synthesize_url,
                headers=headers,
                cookies=cookies,
                json=payload,
                verify=False
            )
            
            response.raise_for_status()
            agent_tts_response = response.json()
            
            if not agent_tts_response.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"AgentZero TTS error: {agent_tts_response.get('error', 'Unknown error')}"
                )
            
            base64_audio = agent_tts_response.get("audio")
            if base64_audio:
                is_final = i == len(sentences) - 1
                await manager.send_stream_chunk(
                    websocket, 
                    "tts_chunk", 
                    base64_audio, 
                    is_final
                )
                
        await manager.send_stream_chunk(websocket, "tts_complete", "TTS generation complete", True)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Streaming TTS error: {error_msg}")
        await manager.send_stream_chunk(websocket, "tts_error", error_msg, True)

# FastRTC TTS streaming function
async def stream_fastrtc_tts(websocket: WebSocket, text: str):
    """Stream TTS audio using FastRTC Kokoro TTS server"""
    try:
        logger.info(f"Starting FastRTC TTS streaming for text: {text[:100]}...")
        await manager.send_stream_chunk(websocket, "tts_status", "Preparing FastRTC TTS...")

        # Get the TTS server
        from kokoro_tts_server import get_tts_server
        tts_server = await get_tts_server()

        await manager.send_stream_chunk(websocket, "tts_status", "Generating audio with Kokoro...")

        # Stream audio chunks using Kokoro TTS
        chunk_count = 0
        async for base64_audio_chunk in tts_server.text_to_base64_streaming(text):
            if base64_audio_chunk:
                chunk_count += 1
                logger.info(f"Streaming FastRTC TTS chunk {chunk_count} ({len(base64_audio_chunk)} chars)")

                # Send chunk to client (is_final will be determined by the last chunk)
                await manager.send_stream_chunk(
                    websocket,
                    "tts_chunk",
                    base64_audio_chunk,
                    False  # Not final yet, will send final signal separately
                )

                # Small delay for smooth streaming
                await asyncio.sleep(0.05)

        # Send completion signal
        await manager.send_stream_chunk(websocket, "tts_complete", "FastRTC TTS generation complete", True)
        logger.info(f"FastRTC TTS streaming completed. Total chunks: {chunk_count}")

    except Exception as e:
        error_msg = f"FastRTC TTS streaming error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await manager.send_stream_chunk(websocket, "tts_error", error_msg, True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await manager.connect(websocket)
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "query":
                context = message.get("context", "")
                request_tts = message.get("request_tts", False)
                logger.info(f"Received WebSocket query: {context}")
                logger.info(f"TTS requested: {request_tts}")

                # Stream the response
                result_text = await stream_agent_response(websocket, context)

                # If TTS was requested, stream audio using FastRTC
                if request_tts and result_text and FASTRTC_AVAILABLE:
                    logger.info("Streaming TTS audio using FastRTC...")
                    await stream_fastrtc_tts(websocket, result_text)
                elif request_tts and result_text:
                    logger.info("FastRTC not available, using AgentZero TTS...")
                    await stream_tts_response(websocket, result_text)
                
            elif message.get("type") == "tts":
                text = message.get("text", "")
                logger.info(f"Received WebSocket TTS request: {text[:100]}...")
                
                # Stream TTS response
                await stream_tts_response(websocket, text)
                
            elif message.get("type") == "ping":
                await manager.send_message(websocket, {"type": "pong", "timestamp": time.time()})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Legacy endpoint - kept for backwards compatibility. Prefer WebSocket streaming.
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
    Transcribe audio using local Faster Whisper or Groq API
    """
    try:
        logger.info("Processing audio transcription request")

        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            # Write the uploaded audio to the temporary file
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        if USE_LOCAL_SERVICES:
            # Use local Faster Whisper STT server
            logger.info("Using local Faster Whisper STT server")
            try:
                with open(temp_audio_path, "rb") as audio_file:
                    files = {"audio": (os.path.basename(temp_audio_path), audio_file, "audio/wav")}
                    response = requests.post(f"{LOCAL_STT_URL}/transcribe", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    transcribed_text = result.get("text", "")
                    logger.info(f"Faster Whisper transcription successful: {transcribed_text}")
                else:
                    logger.error(f"Faster Whisper STT failed: {response.status_code}")
                    raise HTTPException(status_code=500, detail=f"Local STT server error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to local STT server: {e}")
                logger.info("Falling back to Groq API")
                # Fall back to Groq if local server is not available
                transcribed_text = await transcribe_with_groq(temp_audio_path)
        else:
            # Use Groq API
            logger.info("Using Groq API for transcription")
            transcribed_text = await transcribe_with_groq(temp_audio_path)

        # Clean up the temporary file
        os.unlink(temp_audio_path)

        logger.info("Transcription completed successfully")
        return {"text": transcribed_text}
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

async def transcribe_with_groq(temp_audio_path: str) -> str:
    """Helper function to transcribe audio using Groq API"""
    # Get Groq API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")

    # Set up the headers for the Groq API request
    headers = {"Authorization": f"Bearer {groq_api_key}"}

    # Prepare the form data for the Groq API request
    with open(temp_audio_path, "rb") as audio_file:
        files = {"file": (os.path.basename(temp_audio_path), audio_file, "audio/wav")}
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

    # Check if the request was successful
    if response.status_code != 200:
        logger.error(f"Groq API error: {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Groq API error: {response.text}"
        )

    # Parse the response
    result = response.json()
    return result.get("text", "").strip()

@app.post("/speak")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using local Kokoro TTS or AgentZero's TTS API
    """
    try:
        if USE_LOCAL_SERVICES:
            # Use local Kokoro TTS server
            logger.info("Processing text-to-speech request using local Kokoro TTS server")
            try:
                payload = {"text": request.text}
                response = requests.post(f"{LOCAL_TTS_URL}/synthesize", json=payload, timeout=30)

                if response.status_code == 200:
                    logger.info("Kokoro TTS synthesis successful")
                    # Return the audio as a streaming response
                    return StreamingResponse(
                        io.BytesIO(response.content),
                        media_type="audio/wav",
                        headers={"Content-Disposition": "attachment; filename=speech.wav"}
                    )
                else:
                    logger.error(f"Kokoro TTS failed: {response.status_code}")
                    raise HTTPException(status_code=500, detail=f"Local TTS server error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to local TTS server: {e}")
                logger.info("Falling back to AgentZero TTS")
                # Fall back to AgentZero if local server is not available
                return await synthesize_with_agentzero(request.text)
        else:
            # Use AgentZero TTS
            logger.info("Processing text-to-speech request using AgentZero's /synthesize endpoint")
            return await synthesize_with_agentzero(request.text)
        
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
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with performance metrics"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len(context_cache),
        "active_connections": len(manager.active_connections),
        "active_tasks": len(query_tasks),
        "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
    }

# Cache management endpoint
@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    valid_entries = sum(1 for entry in context_cache.values() if is_cache_valid(entry))
    expired_entries = len(context_cache) - valid_entries
    
    return {
        "total_entries": len(context_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_ttl_seconds": cache_ttl
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    cleared_count = len(context_cache)
    context_cache.clear()
    return {
        "message": f"Cleared {cleared_count} cache entries",
        "cleared_count": cleared_count
    }

# FastRTC Integration Endpoints

class ContextUpdateRequest(BaseModel):
    sms_messages: List[Dict[str, Any]] = []
    notifications: List[Dict[str, Any]] = []
    chat_history: List[Dict[str, Any]] = []

class VoiceQueryRequest(BaseModel):
    transcript: str
    context_data: Dict[str, Any] = {}

@app.post("/fastrtc/context")
async def update_fastrtc_context(request: ContextUpdateRequest):
    """Update mobile context for FastRTC voice agent"""
    if not FASTRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="FastRTC not available")
    
    try:
        context_data = {
            "sms_messages": request.sms_messages,
            "notifications": request.notifications,
            "chat_history": request.chat_history,
            "timestamp": time.time()
        }
        
        await update_mobile_context(context_data)
        
        return {
            "status": "success",
            "message": "Context updated successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Context update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fastrtc/voice-query")
async def process_voice_query(request: VoiceQueryRequest):
    """Process voice query directly (alternative to WebRTC for testing)"""
    if not FASTRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="FastRTC not available")
    
    try:
        # Update context if provided
        if request.context_data:
            await update_mobile_context(request.context_data)
        
        # Process voice input
        response = await voice_agent.process_voice_input(request.transcript)
        
        return {
            "status": "success",
            "transcript": request.transcript,
            "response": response,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastrtc/status")
async def fastrtc_status():
    """Get FastRTC system status"""
    status = {
        "fastrtc_available": FASTRTC_AVAILABLE,
        "timestamp": time.time()
    }
    
    if FASTRTC_AVAILABLE:
        try:
            status.update({
                "stt_ready": voice_agent.stt_server is not None,
                "tts_ready": voice_agent.tts_server is not None,
                "agent_initialized": voice_agent.stt_server is not None and voice_agent.tts_server is not None
            })
        except:
            status["agent_initialized"] = False
    
    return status

@app.get("/fastrtc/stream")
async def get_fastrtc_stream_info():
    """Get FastRTC stream configuration for React Native client"""
    if not FASTRTC_AVAILABLE:
        raise HTTPException(status_code=503, detail="FastRTC not available")
    
    try:
        # This would return WebRTC connection details
        # In a full implementation, this would include STUN/TURN servers, etc.
        return {
            "stream_available": True,
            "webrtc_config": {
                "iceServers": [
                    {"urls": "stun:stun.l.google.com:19302"}
                ]
            },
            "endpoints": {
                "context_update": "/fastrtc/context",
                "voice_query": "/fastrtc/voice-query",
                "status": "/fastrtc/status"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Stream info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount FastRTC stream (if available)
if FASTRTC_AVAILABLE:
    try:
        # This would mount the FastRTC stream for WebRTC connections
        # Implementation depends on how FastRTC integrates with FastAPI
        fastrtc_stream = get_fastrtc_stream()
        if fastrtc_stream:
            # fastrtc_stream.mount(app)  # Uncomment when FastRTC is properly configured
            logger.info("FastRTC stream ready for mounting")
    except Exception as e:
        logger.warning(f"FastRTC stream mounting failed: {e}")


# Global start time for uptime tracking
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Proxy Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # Disable reload to avoid multiple server instances
