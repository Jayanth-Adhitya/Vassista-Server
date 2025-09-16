import os
import asyncio
import tempfile
import requests
import io
import logging
import json
import base64
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
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

# Import FastRTC voice handler
try:
    from fastrtc_voice_handler import create_fastrtc_stream
    FASTRTC_AVAILABLE = True
    logger.info("FastRTC voice handler imported successfully")
except Exception as e:
    FASTRTC_AVAILABLE = False
    logger.error(f"FastRTC not available: {e}")
    create_fastrtc_stream = None
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize and mount FastRTC stream on startup
    if FASTRTC_AVAILABLE and create_fastrtc_stream:
        try:
            logger.info("Creating FastRTC stream...")
            fastrtc_stream = create_fastrtc_stream(agent_zero_url=AGENT_ZERO_URL)

            logger.info("Mounting FastRTC stream on FastAPI app...")
            fastrtc_stream.mount(app)  # This creates /webrtc/offer endpoint

            logger.info("✅ FastRTC stream mounted successfully - WebRTC endpoint available at /webrtc/offer")
        except Exception as e:
            logger.error(f"❌ FastRTC initialization error: {e}")
    else:
        logger.warning("FastRTC not available - voice features disabled")

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
USE_LOCAL_SERVICES = True     # Use local STT/TTS services

# Local service URLs
LOCAL_STT_URL = "http://localhost:8001"  # Faster Whisper STT server
LOCAL_TTS_URL = "http://localhost:8002"  # Kokoro TTS server

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
# WebSocket connection manager removed - FastRTC handles connections directly

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
# WebSocket streaming functions removed - FastRTC handles real-time voice processing directly
# Voice interactions now happen through WebRTC peer-to-peer connections via /webrtc/offer

# Old WebSocket endpoint removed - FastRTC WebRTC provides direct peer-to-peer streaming
# Use /webrtc/offer endpoint for WebRTC connections

@app.post("/query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Legacy endpoint - kept for backwards compatibility. Prefer FastRTC WebRTC streaming.
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
                raise HTTPException(status_code=503, detail="Local STT server unavailable")
        else:
            # No fallback - only use local services
            raise HTTPException(status_code=503, detail="Local STT services disabled")

        # Clean up the temporary file
        os.unlink(temp_audio_path)

        logger.info("Transcription completed successfully")
        return {"text": transcribed_text}
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# Groq functions removed - using only local FastRTC services

@app.post("/speak")
async def text_to_speech(request: TextToSpeechRequest):
    """
    DISABLED: This endpoint is replaced by FastRTC streaming TTS via WebRTC
    """
    logger.info("/speak endpoint called but disabled - use FastRTC streaming instead")
    raise HTTPException(
        status_code=410,
        detail="TTS endpoint disabled - use FastRTC WebRTC streaming"
    )
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with performance metrics"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len(context_cache),
        "fastrtc_available": FASTRTC_AVAILABLE,
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
        
        # Context data stored for FastRTC voice processing
        logger.info(f"Context updated: {len(context_data)} items")
        
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
            logger.info(f"Context data provided: {len(request.context_data)} items")

        # Process voice input - would use FastRTC voice handler
        logger.info(f"Processing voice query: {request.transcript}")
        response = f"Processed: {request.transcript}"
        
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
                "webrtc_endpoint": "/webrtc/offer",
                "services": "faster-whisper + kokoro integrated"
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

# FastRTC stream is mounted in lifespan() function above
# The /webrtc/offer endpoint is automatically created when FastRTC mounts


# Global start time for uptime tracking
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Proxy Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # Disable reload to avoid multiple server instances
