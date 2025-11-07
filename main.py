import os
import asyncio
import tempfile
import requests
import io
import logging
import json
import base64
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, Body, WebSocket, Form, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, List, Optional
import uuid

# MCP SSE imports
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import voice handler components directly
try:
    from faster_whisper_stt_server import FasterWhisperSTTServer
    from kokoro_tts_server import KokoroTTSServer
    VOICE_SERVICES_AVAILABLE = True
    logger.info("Voice services imported successfully")
except ImportError as e:
    # Try to fix the backports issue
    import sys
    import importlib
    if 'backports' in str(e):
        # Remove backports from sys.modules to force reimport
        if 'backports.tarfile' in sys.modules:
            del sys.modules['backports.tarfile']
        if 'backports' in sys.modules:
            del sys.modules['backports']
    try:
        from faster_whisper_stt_server import FasterWhisperSTTServer
        from kokoro_tts_server import KokoroTTSServer
        VOICE_SERVICES_AVAILABLE = True
        logger.info("Voice services imported successfully after fixing backports")
    except Exception as e2:
        VOICE_SERVICES_AVAILABLE = False
        logger.error(f"Voice services not available: {e2}")
        FasterWhisperSTTServer = None
        KokoroTTSServer = None
except Exception as e:
    VOICE_SERVICES_AVAILABLE = False
    logger.error(f"Voice services not available: {e}")
    FasterWhisperSTTServer = None
    KokoroTTSServer = None

# Import SMS RAG components
try:
    from sms_rag import SMSVectorStore, SMSQueryAnalyzer
    SMS_RAG_AVAILABLE = True
    logger.info("SMS RAG services imported successfully")
except ImportError as e:
    SMS_RAG_AVAILABLE = False
    logger.error(f"SMS RAG services not available: {e}")
    SMSVectorStore = None
    SMSQueryAnalyzer = None

# WebRTC signaling server
from webrtc_signaling import signaling_server
from fastrtc_voice_agent import initialize_audio_relay
import asyncio
import json
import numpy as np
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize voice services on startup
    global stt_server, tts_server, sms_store, query_analyzer
    stt_server = None
    tts_server = None
    sms_store = None
    query_analyzer = None

    if VOICE_SERVICES_AVAILABLE:
        try:
            logger.info("Initializing voice services...")

            # Initialize STT server
            stt_server = FasterWhisperSTTServer(
                model_name="tiny",  # Fast model for real-time
                device="auto",
                compute_type="int8"
            )
            await stt_server.initialize()
            logger.info("✅ Faster Whisper STT initialized")

            # Initialize TTS server
            tts_server = KokoroTTSServer(
                device="auto",
                sample_rate=24000
            )
            tts_initialized = await tts_server.initialize()
            if tts_initialized:
                logger.info("✅ Kokoro TTS initialized")
            else:
                logger.error("❌ Kokoro TTS initialization failed")
                tts_server = None

            # Initialize audio relay
            audio_relay = initialize_audio_relay(stt_server, tts_server)
            logger.info("✅ FastRTC voice agent initialized")
            logger.info(f"🎵 Audio relay TTS server: {audio_relay.tts_server is not None}")
            logger.info(f"🎵 Audio relay STT server: {audio_relay.stt_server is not None}")

            logger.info("✅ Voice services initialized successfully")
            logger.info("✅ WebRTC signaling server ready at /ws")
        except Exception as e:
            logger.error(f"❌ Voice services initialization error: {e}")
            stt_server = None
            tts_server = None
    else:
        logger.warning("Voice services not available - WebRTC signaling only")
        # Initialize audio relay without voice services (for basic signaling)
        audio_relay = initialize_audio_relay(None, None)
        logger.info("✅ FastRTC voice agent initialized (signaling only)")
        logger.info("✅ WebRTC signaling server ready at /ws")

    # Initialize SMS RAG services
    if SMS_RAG_AVAILABLE:
        try:
            logger.info("Initializing SMS RAG services...")
            sms_store = SMSVectorStore(
                persist_directory="./chroma_sms_db",
                collection_name="sms_messages",
                model_name="all-MiniLM-L6-v2"
            )
            query_analyzer = SMSQueryAnalyzer()
            logger.info("✅ SMS RAG services initialized successfully")
        except Exception as e:
            logger.error(f"❌ SMS RAG services initialization error: {e}")
            sms_store = None
            query_analyzer = None
    else:
        logger.warning("SMS RAG services not available")

    yield

# Global voice service instances
stt_server = None
tts_server = None
# Global SMS RAG service instances
sms_store = None
query_analyzer = None
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

# ================ MCP SSE Server Setup ================
# Create FastMCP server for SMS tools
mcp_server = FastMCP("sms-search")

@mcp_server.tool()
async def search_sms(query: str, top_k: int = 10, time_window_days: int = None, contact: str = None) -> str:
    """
    Search through SMS messages semantically using vector similarity.

    Args:
        query: The search query (e.g., "money received", "doctor appointment")
        top_k: Number of results to return (default: 10)
        time_window_days: Filter messages from last N days (optional)
        contact: Filter by contact name/number (optional)

    Returns:
        Formatted SMS results with date, contact, and message content
    """
    logger.info(f"🔧 MCP Tool: search_sms - query='{query}', top_k={top_k}")

    result = await execute_sms_tool("search_sms", {
        "query": query,
        "top_k": top_k,
        "time_window_days": time_window_days,
        "contact": contact
    })

    if "error" in result:
        return f"Error: {result['error']}"

    results = result.get("results", [])
    if not results:
        return "No SMS messages found matching your query."

    return format_sms_results_for_context(results)

@mcp_server.tool()
async def get_recent_sms(limit: int = 10, days: int = 7, contact: str = None) -> str:
    """
    Get recent SMS messages chronologically.

    Args:
        limit: Maximum number of messages to return (default: 10)
        days: Look back N days (default: 7)
        contact: Filter by contact name/number (optional)

    Returns:
        Formatted list of recent SMS messages
    """
    logger.info(f"🔧 MCP Tool: get_recent_sms - limit={limit}, days={days}")

    result = await execute_sms_tool("get_recent_sms", {
        "limit": limit,
        "days": days,
        "contact": contact
    })

    if "error" in result:
        return f"Error: {result['error']}"

    results = result.get("results", [])
    if not results:
        return f"No SMS messages found in the last {days} days."

    return format_sms_results_for_context(results)

@mcp_server.tool()
async def count_sms(days: int = None, contact: str = None) -> str:
    """
    Count total SMS messages.

    Args:
        days: Count from last N days (optional, counts all if not specified)
        contact: Filter by contact name/number (optional)

    Returns:
        Total count of SMS messages matching filters
    """
    logger.info(f"🔧 MCP Tool: count_sms - days={days}, contact={contact}")

    result = await execute_sms_tool("count_sms", {
        "days": days,
        "contact": contact
    })

    if "error" in result:
        return f"Error: {result['error']}"

    count = result.get("count", 0)
    time_str = f" from the last {days} days" if days else ""
    contact_str = f" from {contact}" if contact else ""
    return f"Total SMS messages{time_str}{contact_str}: {count}"

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
FASTRTC_AVAILABLE = True      # FastRTC WebRTC components available

# Local service URLs
LOCAL_STT_URL = "http://localhost:8001"  # Faster Whisper STT server
LOCAL_TTS_URL = "http://localhost:8002"  # Kokoro TTS server

# AgentZero external services (for AI responses, not TTS)
AGENT_ZERO_URL = "https://ao.uptopoint.net"
# Request models
class QueryRequest(BaseModel):
    context: str
    includeMessages: Optional[bool] = False
    includeNotifications: Optional[bool] = False
    includeChatHistory: Optional[bool] = False
    notifications: Optional[List[Dict]] = []
    chatHistory: Optional[List[Dict]] = []

class QueryResponse(BaseModel):
    result: str
class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "Fritz-PlayAI"  # Default voice
class TranscriptionResponse(BaseModel):
    text: str

# SMS RAG request/response models
class SMSIndexRequest(BaseModel):
    messages: List[Dict]
    time_window_days: Optional[int] = 30  # Default to 30 days

class SMSSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    time_window_days: Optional[int] = None
    contact: Optional[str] = None

class SmartQueryRequest(BaseModel):
    message: str
    include_sms: bool = False
    sms_messages: Optional[List[Dict]] = None
    include_notifications: bool = False
    include_chat_history: bool = False

class SmartQueryResponse(BaseModel):
    response: str
    context_used: Dict
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
    Text chat endpoint with SMS semantic search support.
    Now supports includeMessages flag for automatic SMS context retrieval.
    """
    task_id = str(uuid.uuid4())
    query_tasks[task_id] = {"status": "running", "result": None}

    # Log the received request for debugging
    logger.info(f"📝 Received text query request: {req.context[:100]}...")
    logger.info(f"🔍 includeMessages: {req.includeMessages}")

    async def run_task():
        try:
            # If SMS context is requested, use generate_ai_response() with semantic search
            if req.includeMessages and SMS_RAG_AVAILABLE and sms_store:
                logger.info("🔍 Using SMS semantic search for text query")

                # Build context data
                context_data = {
                    'includeMessages': req.includeMessages,
                    'includeNotifications': req.includeNotifications,
                    'includeChatHistory': req.includeChatHistory,
                    'notifications': req.notifications or [],
                    'chatHistory': req.chatHistory or []
                }

                # Use generate_ai_response which includes SMS semantic search
                result_text = await generate_ai_response(req.context, "default", context_data)

                query_tasks[task_id] = {"status": "completed", "result": result_text}
                return

            # Fallback to direct AgentZero call (no SMS search)
            logger.info("📤 Using direct AgentZero call (no SMS search)")

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

            # Generate TTS audio for the response
            tts_audio_base64 = None
            if tts_server:
                try:
                    logger.info(f"🎵 Generating TTS for response: {result_text[:50]}...")
                    audio_data = await tts_server.synthesize(result_text)
                    if len(audio_data) > 0:
                        # Convert numpy array to bytes - handle different data types properly
                        if audio_data.dtype == np.float32:
                            # Convert float32 to int16
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()
                        elif audio_data.dtype == np.int16:
                            # Already int16, just convert to bytes
                            audio_bytes = audio_data.tobytes()
                        else:
                            # Convert to int16 first
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()

                        # Ensure we have valid audio data
                        if len(audio_bytes) > 0:
                            # Validate audio data before encoding
                            logger.info(f"🔊 Audio data: dtype={audio_data.dtype}, shape={audio_data.shape}, bytes={len(audio_bytes)}")

                            # Ensure audio data is not corrupted
                            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                                logger.error("❌ Audio data contains NaN or Inf values")
                                tts_audio_base64 = None
                            else:
                                tts_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                                logger.info(f"✅ TTS audio generated: {len(audio_bytes)} bytes, {len(tts_audio_base64)} chars")
                        else:
                            logger.warning("⚠️ TTS generated empty audio bytes")
                            tts_audio_base64 = None
                    else:
                        logger.warning("⚠️ TTS returned empty audio")
                except Exception as e:
                    logger.error(f"❌ TTS generation error: {e}")

            # Store result with TTS audio
            query_tasks[task_id] = {
                "status": "completed",
                "result": result_text,
                "tts_audio": tts_audio_base64
            }

            # Send TTS audio back to client via audio relay (for all requests)
            try:
                if tts_audio_base64:
                    # Use audio relay to send TTS response
                    from fastrtc_voice_agent import audio_relay
                    if audio_relay:
                        # Send TTS response through audio relay
                        tts_bytes = base64.b64decode(tts_audio_base64)
                        await audio_relay._send_tts_response("default", result_text, tts_bytes)
                        logger.info("📤 Sent TTS audio response via audio relay")
            except Exception as e:
                logger.error(f"❌ Error sending TTS response: {e}")
        except Exception as e:
            error_message = str(e)

            # Check for rate limit errors and provide user-friendly message
            if ('RateLimitError' in error_message or
                '429' in error_message or
                'Too Many Requests' in error_message or
                'exceeded your current quota' in error_message or
                'RESOURCE_EXHAUSTED' in error_message):

                user_friendly_error = "Rate limit reached. Please wait a moment before sending another message."
                logger.warning(f"Rate limit detected: {error_message[:200]}...")
            else:
                user_friendly_error = f"Error: {error_message}"
                logger.error(f"Query processing error: {e}")

            query_tasks[task_id] = {"status": "error", "result": user_friendly_error}
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

@app.get("/query_result_with_tts/{task_id}")
async def get_query_result_with_tts(task_id: str):
    """
    Get query result with TTS audio included.
    """
    task = query_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # If TTS audio is not already generated, generate it now
    if task.get("status") == "completed" and "tts_audio" not in task and tts_server:
        try:
            result_text = task.get("result", "")
            logger.info(f"🎵 Generating TTS for existing result: {result_text[:50]}...")

            audio_data = await tts_server.synthesize(result_text)
            if len(audio_data) > 0:
                # Convert numpy array to bytes
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                tts_audio_base64 = base64.b64encode(audio_bytes).decode()
                task["tts_audio"] = tts_audio_base64
                logger.info("✅ TTS audio generated for existing result")
            else:
                logger.warning("⚠️ TTS returned empty audio for existing result")
        except Exception as e:
            logger.error(f"❌ TTS generation error for existing result: {e}")

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

@app.post("/upload-audio")
async def upload_audio_file(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    room: str = Query("default", description="Room identifier"),
    client_id: str = Query(None, description="Client identifier"),
    session_id: str = Query(None, description="Session identifier"),
    context_data: str = Form(None, description="Context data JSON string")
):
    """
    Upload audio file for real-time processing using HTTP FormData.
    This endpoint processes uploaded audio files with Faster Whisper STT
    and returns results via WebSocket connection.

    - **audio_file**: Audio file (.wav format preferred)
    - **room**: Room identifier for WebSocket response routing
    - **client_id**: Client identifier for response targeting
    - **session_id**: Session identifier for tracking
    """
    try:
        logger.info("🎵 ========== AUDIO FILE UPLOAD RECEIVED ==========")
        logger.info(f"📁 File: {audio_file.filename}")
        logger.info(f"📊 Content type: {audio_file.content_type}")
        logger.info(f"📏 File size: {audio_file.size} bytes")
        logger.info(f"🏠 Room: {room}")
        logger.info(f"👤 Client ID: {client_id}")
        logger.info(f"🔗 Session ID: {session_id}")

        # Parse context data
        parsed_context_data = {}
        if context_data:
            try:
                import json
                parsed_context_data = json.loads(context_data)
                logger.info(f"📊 Context data received: {len(parsed_context_data.get('notifications', []))} notifications, {len(parsed_context_data.get('chatHistory', []))} chat messages, includeMessages={parsed_context_data.get('includeMessages', False)}")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Failed to parse context data: {e}")
                parsed_context_data = {}

        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if audio_file.size > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        # Validate file type
        allowed_types = ['audio/wav', 'audio/wave', 'audio/mpeg', 'audio/mp4', 'audio/webm']
        if audio_file.content_type not in allowed_types:
            logger.warning(f"⚠️ Unsupported audio type: {audio_file.content_type}")

        # Create temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{room}_{int(time.time())}.wav")
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            # Save uploaded file
            content = await audio_file.read()
            with open(temp_file_path, 'wb') as f:
                f.write(content)

            logger.info(f"💾 Saved audio file: {temp_file_path}")
            logger.info(f"📊 Saved file size: {len(content)} bytes")

            # Process audio file in background
            background_tasks.add_task(
                process_uploaded_audio_file,
                temp_file_path,
                room,
                client_id,
                session_id,
                parsed_context_data
            )

            return {
                "status": "processing",
                "message": "Audio file uploaded successfully, processing started",
                "file_path": temp_file_path,
                "room": room,
                "client_id": client_id,
                "session_id": session_id,
                "timestamp": time.time()
            }

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_file_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

async def process_uploaded_audio_file(file_path: str, room: str, client_id: str, session_id: str, context_data: dict = None):
    """
    Process uploaded audio file with STT and return results via WebSocket
    """
    try:
        logger.info(f"🎵 ========== PROCESSING AUDIO FILE ==========")
        logger.info(f"📁 File: {file_path}")
        logger.info(f"🏠 Room: {room}")
        logger.info(f"👤 Client: {client_id}")

        # Check if file exists and get size
        if not os.path.exists(file_path):
            logger.error(f"❌ Audio file not found: {file_path}")
            return

        file_size = os.path.getsize(file_path)
        logger.info(f"📊 File size: {file_size} bytes")

        # Process with STT server
        transcript = await process_audio_with_stt(file_path)

        if not transcript or not transcript.strip():
            logger.warning("⚠️ No transcript generated from audio file")
            transcript = "I couldn't understand the audio. Please try again."

        logger.info(f"📝 Final transcript: {transcript}")

        # Generate AI response with context
        ai_response = await generate_ai_response(transcript, room, parsed_context_data)

        # Generate streaming TTS audio
        await generate_streaming_tts_audio(ai_response, room, client_id, transcript, session_id)

        logger.info("✅ Audio file processing completed successfully")

    except Exception as e:
        logger.error(f"❌ Error processing audio file: {e}")

        # Send error response via WebSocket
        try:
            await send_websocket_response(room, client_id, {
                "type": "error",
                "error": f"Audio processing failed: {str(e)}",
                "session_id": session_id,
                "timestamp": time.time()
            })
        except:
            logger.error("❌ Failed to send error response via WebSocket")

    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error cleaning up file {file_path}: {e}")

async def process_audio_with_stt(file_path: str) -> str:
    """
    Process audio file with Faster Whisper STT
    """
    try:
        logger.info("🎵 Processing audio file with STT...")

        # Use local STT server if available
        if stt_server:
            logger.info("🎯 Using local STT server")

            # Load audio file as numpy array
            import soundfile as sf
            audio_data, sample_rate = sf.read(file_path)

            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0

            # Transcribe with STT server
            transcript = await stt_server.transcribe_audio_array(audio_data, sample_rate=sample_rate)

            if transcript and transcript.strip():
                logger.info(f"✅ STT transcription successful: {transcript}")
                return transcript
            else:
                logger.warning("⚠️ STT returned empty transcript")
                return None

        else:
            logger.error("❌ STT server not available")
            return "Speech-to-text service is not available."

    except Exception as e:
        logger.error(f"❌ STT processing error: {e}")
        return f"Error processing audio: {str(e)}"

async def generate_ai_response(transcript: str, room: str, context_data: dict = None) -> str:
    """
    Generate AI response to transcript with context data, now with SMS RAG support
    """
    try:
        logger.info(f"🤖 Generating AI response to: {transcript[:50]}...")
        logger.info(f"📋 Context data received: {context_data}")

        # Use SmartQueryRequest for enhanced context processing if SMS RAG is available
        if SMS_RAG_AVAILABLE and sms_store and query_analyzer and context_data and context_data.get('includeMessages'):
            try:
                # Create smart query request
                smart_request = SmartQueryRequest(
                    message=transcript,
                    include_sms=True,
                    sms_messages=context_data.get('smsMessages', []),
                    include_notifications=context_data.get('includeNotifications', False),
                    include_chat_history=context_data.get('includeChatHistory', False)
                )

                # Use smart query processing
                result = await smart_query(smart_request)
                if result.response and result.response.strip():
                    logger.info(f"✅ AI response generated with SMS RAG: {result.response[:50]}...")
                    return result.response
            except Exception as smart_query_error:
                logger.warning(f"⚠️ Smart query failed, falling back to legacy method: {smart_query_error}")

        # Fallback to original context building method
        context_parts = []

        # Start with the user query
        context_parts.append(f"User Query: {transcript}")

        # Add context information if available
        has_context = False

        if context_data:
            # Add notifications context
            if context_data.get('notifications') and len(context_data['notifications']) > 0:
                has_context = True
                context_parts.append("\n--- Recent App Notifications ---")
                for i, notif in enumerate(context_data['notifications'], 1):
                    app_name = notif.get('packageName', 'Unknown').split('.')[-1].capitalize()
                    title = notif.get('title', 'No Title')
                    body = notif.get('body', 'No Body')
                    context_parts.append(f"{i}. {app_name}: {title}")
                    if body and body != title:
                        context_parts.append(f"   Content: {body}")
                logger.info(f"📱 Added {len(context_data['notifications'])} notifications to context")

            # Add chat history context
            if context_data.get('chatHistory') and len(context_data['chatHistory']) > 0:
                has_context = True
                context_parts.append("\n--- Recent Chat History ---")
                for msg in context_data['chatHistory']:
                    sender = "User" if msg.get('sender') == 'user' else "Assistant"
                    text = msg.get('text', '')
                    context_parts.append(f"{sender}: {text}")
                logger.info(f"💬 Added {len(context_data['chatHistory'])} chat messages to context")

            # Add SMS messages context using semantic search if available
            logger.info(f"🔍 SMS Context Check: includeMessages={context_data.get('includeMessages') if context_data else None}, SMS_RAG_AVAILABLE={SMS_RAG_AVAILABLE}, sms_store={'exists' if sms_store else 'None'}")

            if context_data.get('includeMessages'):
                if SMS_RAG_AVAILABLE and sms_store:
                    try:
                        # Use semantic search to find relevant SMS messages
                        logger.info(f"🔍 Performing semantic SMS search for: {transcript[:50]}...")
                        sms_results = await asyncio.to_thread(
                            sms_store.search_sms,
                            transcript,
                            top_k=5,  # Only get top 5 most relevant messages
                            time_window_days=30  # Last 30 days
                        )

                        if sms_results:
                            has_context = True
                            context_parts.append("\n--- Relevant SMS Messages (Semantic Search) ---")
                            formatted_sms = format_sms_results_for_context(sms_results)
                            context_parts.append(formatted_sms)
                            logger.info(f"📱 Added {len(sms_results)} semantically relevant SMS messages to context")
                        else:
                            context_parts.append("\n--- Relevant SMS Messages ---")
                            context_parts.append("No relevant SMS messages found in your message history.")
                            logger.info("📱 Semantic SMS search returned no results")
                    except Exception as sms_error:
                        logger.error(f"SMS semantic search error: {sms_error}")
                        context_parts.append("\n--- SMS Messages ---")
                        context_parts.append("SMS search temporarily unavailable.")
                else:
                    # Fallback to provided SMS messages if vector store not available
                    sms_messages = context_data.get('smsMessages', [])
                    if sms_messages:
                        has_context = True
                        context_parts.append("\n--- Recent SMS Messages (Chronological) ---")
                        for i, sms in enumerate(sms_messages, 1):
                            sender = sms.get('address', 'Unknown')
                            body = sms.get('body', '')
                            date = sms.get('date', '')
                            context_parts.append(f"{i}. From: {sender}")
                            context_parts.append(f"   Message: {body}")
                            if date:
                                context_parts.append(f"   Time: {date}")
                        logger.info(f"📱 Added {len(sms_messages)} SMS messages to context (fallback mode)")
                    else:
                        context_parts.append("\n--- Recent SMS Messages ---")
                        context_parts.append("No recent SMS messages available.")
                        logger.info("📱 SMS messages requested but none available")

        # Add instruction for context-aware responses
        if has_context:
            context_parts.append("\n--- Instructions ---")
            context_parts.append("Please respond to the user query while considering the context information provided above. Be helpful and reference relevant context when appropriate.")

        full_context = '\n'.join(context_parts)

        logger.info(f"📝 Built context prompt with {len(full_context)} characters")
        logger.info(f"📝 Context preview: {full_context[:300]}...")

        # Use AgentZero for AI response with full context
        ai_response = await call_agent_zero(full_context, is_context_enhanced=True)

        if ai_response and ai_response.strip():
            logger.info(f"✅ AI response generated: {ai_response[:50]}...")
            return ai_response
        else:
            logger.warning("⚠️ AI returned empty response")
            return "I'm here to help! Could you please repeat that?"

    except Exception as e:
        logger.error(f"❌ AI response generation error: {e}")
        return "I'm sorry, I encountered an error processing your request."

async def generate_tts_audio(text: str) -> str:
    """
    Generate TTS audio from text (batch mode - kept for compatibility)
    """
    try:
        if not tts_server:
            logger.warning("⚠️ TTS server not available")
            return ""

        logger.info(f"🔊 Generating TTS audio for: {text[:50]}...")

        # Generate TTS audio
        audio_data = await tts_server.synthesize(text)

        if len(audio_data) > 0:
            # Convert numpy array to bytes
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()

            # Convert to base64 for transmission
            tts_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            logger.info(f"✅ TTS audio generated: {len(tts_audio_base64)} characters")
            return tts_audio_base64
        else:
            logger.warning("⚠️ TTS returned empty audio")
            return ""

    except Exception as e:
        logger.error(f"❌ TTS generation error: {e}")
        return ""

async def generate_streaming_tts_audio(text: str, room: str, client_id: str, transcript: str, session_id: str):
    """
    Generate streaming TTS audio from text
    """
    try:
        if not tts_server:
            logger.warning("⚠️ TTS server not available")
            return

        # Skip TTS for error messages to prevent streaming conflicts
        error_indicators = [
            "I'm sorry, there was an error",
            "I'm temporarily busy",
            "I'm receiving too many requests",
            "The request took too long",
            "Please try again"
        ]

        if any(indicator in text for indicator in error_indicators):
            logger.info("⚠️ Skipping TTS for error message to prevent streaming conflicts")
            # Send simple response without TTS
            await send_websocket_response(room, client_id, {
                "type": "response",
                "text": text,
                "transcript": transcript,
                "audio": False,  # No TTS audio for error messages
                "format": "text_only",
                "session_id": session_id,
                "timestamp": time.time(),
                "error_response": True
            })
            logger.info(f"📤 Error response sent to room {room}")
            return

        logger.info(f"🎵 Starting streaming TTS for: {text[:50]}...")

        # Send response first so client can display the text immediately
        await send_websocket_response(room, client_id, {
            "type": "response",
            "text": text,
            "transcript": transcript,
            "audio": True,  # Indicates TTS audio is coming
            "format": "streaming_wav",
            "sample_rate": 24000,
            "session_id": session_id,
            "timestamp": time.time()
        })
        logger.info(f"📤 Response sent to room {room}")

        # Send TTS start event
        await send_websocket_response(room, client_id, {
            "type": "tts_start",
            "data": {
                "text": text,
                "transcript": transcript,
                "total_text_length": len(text),
                "session_id": session_id,
                "timestamp": time.time()
            }
        })
        logger.info(f"📤 TTS streaming started for room {room}")

        # Stream TTS chunks
        chunk_count = 0

        try:
            # Use the streaming TTS functionality
            async for audio_base64_chunk in tts_server.text_to_base64_streaming(text):
                if audio_base64_chunk and len(audio_base64_chunk) > 0:
                    chunk_count += 1

                    # Send TTS chunk
                    await send_websocket_response(room, client_id, {
                        "type": "tts_chunk",
                        "data": {
                            "audio": audio_base64_chunk,
                            "format": "wav",
                            "sample_rate": 24000,
                            "chunk_index": chunk_count,
                            "is_final": False,
                            "session_id": session_id,
                            "timestamp": time.time()
                        }
                    })
                    logger.info(f"📤 TTS chunk {chunk_count} sent to room {room} ({len(audio_base64_chunk)} chars)")

                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)

            # Send TTS complete event
            await send_websocket_response(room, client_id, {
                "type": "tts_complete",
                "data": {
                    "text": text,
                    "transcript": transcript,
                    "total_chunks": chunk_count,
                    "success": True,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            })
            logger.info(f"✅ TTS streaming completed for room {room} ({chunk_count} chunks)")

        except Exception as streaming_error:
            logger.error(f"❌ TTS streaming error: {streaming_error}")

            # Send error message
            await send_websocket_response(room, client_id, {
                "type": "tts_error",
                "data": {
                    "error": str(streaming_error),
                    "chunks_sent": chunk_count,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            })

    except Exception as e:
        logger.error(f"❌ Streaming TTS generation error: {e}")

        # Fallback to transcript-only response
        await send_websocket_response(room, client_id, {
            "type": "response",
            "transcript": transcript,
            "text": text,
            "audio": "",
            "format": "wav",
            "sample_rate": 24000,
            "session_id": session_id,
            "timestamp": time.time(),
            "error": "TTS streaming failed"
        })

async def send_websocket_response(room: str, client_id: str, message: dict):
    """
    Send response back via WebSocket to the appropriate room/client
    """
    try:
        logger.info(f"📡 Sending WebSocket response to room: {room}, client: {client_id}")

        # Use the signaling server to broadcast to the room
        from webrtc_signaling import signaling_server

        # Add sender information
        message.update({
            "sender_id": "server",
            "room": room,
            "client_id": client_id
        })

        # Broadcast to room
        await signaling_server.broadcast_to_room(room, message)

        logger.info("✅ WebSocket response sent successfully")

    except Exception as e:
        logger.error(f"❌ Error sending WebSocket response: {e}")
        logger.error(f"❌ Room: {room}, Client: {client_id}")

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
        "voice_services_available": VOICE_SERVICES_AVAILABLE,
        "stt_initialized": stt_server is not None,
        "tts_initialized": tts_server is not None,
        "sms_rag_available": SMS_RAG_AVAILABLE,
        "sms_store_initialized": sms_store is not None,
        "query_analyzer_initialized": query_analyzer is not None,
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

# SMS RAG Endpoints
@app.post("/index-sms")
async def index_sms_messages(request: SMSIndexRequest):
    """
    Index SMS messages for semantic search with time window filtering.

    Args:
        request: Contains messages list and time_window_days (default 30)

    Returns:
        Indexing statistics (indexed, skipped, errors, total)
    """
    if not SMS_RAG_AVAILABLE or not sms_store:
        raise HTTPException(status_code=503, detail="SMS RAG services not available")

    try:
        logger.info(f"Indexing {len(request.messages)} SMS messages with {request.time_window_days} day window")

        # Index messages in background thread
        result = await asyncio.to_thread(
            sms_store.add_sms_batch,
            request.messages,
            request.time_window_days
        )

        logger.info(f"SMS indexing complete: {result}")
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        logger.error(f"SMS indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-query")
async def smart_query(request: SmartQueryRequest):
    """Process query with intelligent SMS context retrieval"""
    try:
        context_parts = []
        context_metadata = {}

        # Analyze if SMS context is needed
        if request.include_sms and query_analyzer and sms_store and query_analyzer.needs_sms_context(request.message):
            # Use provided SMS messages or get from database
            sms_context = ""
            if request.sms_messages:
                # Index provided SMS messages first
                await asyncio.to_thread(sms_store.add_sms_batch, request.sms_messages)

            # Get relevant SMS context via semantic search
            sms_context = await asyncio.to_thread(
                sms_store.get_relevant_context,
                request.message,
                max_tokens=1500
            )

            # Extract query filters
            filters = query_analyzer.extract_query_filters(request.message)

            if sms_context:
                context_parts.append(f"Relevant SMS Messages:\n{sms_context}")
                context_metadata['sms_included'] = True
                context_metadata['sms_filters'] = filters

        # Build final prompt with context
        final_prompt = request.message
        if context_parts:
            final_prompt = f"Context:\n{chr(10).join(context_parts)}\n\nUser Query: {request.message}"

        logger.info(f"Smart query prompt length: {len(final_prompt)}")

        # Call AgentZero AI with enhanced context
        ai_response = await call_agent_zero(final_prompt, is_context_enhanced=True)

        return SmartQueryResponse(
            response=ai_response,
            context_used=context_metadata
        )

    except Exception as e:
        logger.error(f"Smart query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sms-search")
async def search_sms_messages(request: SMSSearchRequest):
    """
    Direct semantic search endpoint for SMS with filters.

    Args:
        request: Contains query, top_k, time_window_days, contact filters

    Returns:
        List of matching SMS messages with similarity scores
    """
    if not SMS_RAG_AVAILABLE or not sms_store:
        raise HTTPException(status_code=503, detail="SMS RAG services not available")

    try:
        results = await asyncio.to_thread(
            sms_store.search_sms,
            request.query,
            top_k=request.top_k,
            time_window_days=request.time_window_days,
            contact=request.contact
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"SMS search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sms-recent")
async def get_recent_sms_messages(
    limit: int = 10,
    contact: Optional[str] = None,
    days: int = 7
):
    """
    Get recent SMS messages (chronological order).

    Args:
        limit: Max number of messages to return
        contact: Optional contact filter
        days: Number of days to look back (default 7)

    Returns:
        List of recent SMS messages
    """
    if not SMS_RAG_AVAILABLE or not sms_store:
        raise HTTPException(status_code=503, detail="SMS RAG services not available")

    try:
        results = await asyncio.to_thread(
            sms_store.get_recent_sms,
            limit=limit,
            contact=contact,
            days=days
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"SMS recent messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sms-count")
async def count_sms_messages(
    contact: Optional[str] = None,
    days: Optional[int] = None
):
    """
    Count total SMS messages in the store.

    Args:
        contact: Optional contact filter
        days: Optional time window in days

    Returns:
        Count of SMS messages
    """
    if not SMS_RAG_AVAILABLE or not sms_store:
        raise HTTPException(status_code=503, detail="SMS RAG services not available")

    try:
        count = await asyncio.to_thread(
            sms_store.count_sms,
            contact=contact,
            days=days
        )
        return {"count": count}
    except Exception as e:
        logger.error(f"SMS count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================ MCP SSE Endpoints for Remote AgentZero ================

# Mount FastMCP's SSE app at /mcp endpoint
# This provides both /mcp/sse (for SSE connections) and /mcp/messages/ (for POST messages)
logger.info("🔌 Mounting MCP SSE server at /mcp")
app.mount("/mcp", mcp_server.sse_app())

# ================ Legacy HTTP MCP Endpoints (Not Used - FastMCP handles via SSE) ================
# These endpoints are kept for reference but are not active since /mcp is mounted to FastMCP SSE server
# To use these, you would need to mount them at a different path like /api/mcp/tools

# @app.get("/mcp/tools") - DISABLED: Conflicts with FastMCP mount
@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """
    List all available MCP tools for AgentZero.
    This endpoint allows AgentZero to discover what SMS tools are available.
    """
    tools = get_sms_tools_schema()

    return {
        "tools": tools,
        "server_info": {
            "name": "sms-search-server",
            "version": "1.0.0",
            "description": "SMS search and analysis tools",
            "capabilities": ["tools"]
        }
    }

class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

@app.post("/api/mcp/call-tool")
async def call_mcp_tool(request: MCPToolCallRequest):
    """
    Execute an MCP tool call from AgentZero.
    AgentZero can call this endpoint to execute SMS search tools.
    """
    logger.info(f"🔧 MCP Tool Call: {request.tool_name} with args: {request.arguments}")

    if not SMS_RAG_AVAILABLE or not sms_store:
        raise HTTPException(status_code=503, detail="SMS RAG services not available")

    try:
        tool_name = request.tool_name
        arguments = request.arguments

        if tool_name == "search_sms":
            query = arguments.get("query")
            if not query:
                raise HTTPException(status_code=400, detail="Query parameter is required")

            top_k = arguments.get("top_k", 10)
            time_window_days = arguments.get("time_window_days")
            contact = arguments.get("contact")

            results = await asyncio.to_thread(
                sms_store.search_sms,
                query,
                top_k=top_k,
                time_window_days=time_window_days,
                contact=contact
            )

            if not results:
                return {
                    "success": True,
                    "result": "No relevant SMS messages found."
                }

            # Format results
            formatted_results = []
            for i, msg in enumerate(results, 1):
                date_iso = msg.get('date_iso', '')
                address = msg.get('address', 'unknown')
                body = msg.get('body', '')
                similarity = msg.get('similarity', 0)

                formatted_results.append(
                    f"{i}. [{date_iso}] From: {address} (relevance: {similarity:.2f})\n"
                    f"   Message: {body}"
                )

            result_text = "\n\n".join(formatted_results)
            return {
                "success": True,
                "result": f"Found {len(results)} relevant SMS messages:\n\n{result_text}"
            }

        elif tool_name == "get_recent_sms":
            limit = arguments.get("limit", 10)
            contact = arguments.get("contact")
            days = arguments.get("days", 7)

            results = await asyncio.to_thread(
                sms_store.get_recent_sms,
                limit=limit,
                contact=contact,
                days=days
            )

            if not results:
                return {
                    "success": True,
                    "result": "No recent SMS messages found."
                }

            # Format results
            formatted_results = []
            for i, msg in enumerate(results, 1):
                date_iso = msg.get('date_iso', '')
                address = msg.get('address', 'unknown')
                body = msg.get('body', '')

                formatted_results.append(
                    f"{i}. [{date_iso}] From: {address}\n"
                    f"   Message: {body}"
                )

            result_text = "\n\n".join(formatted_results)
            return {
                "success": True,
                "result": f"Recent SMS messages (last {days} days):\n\n{result_text}"
            }

        elif tool_name == "count_sms":
            contact = arguments.get("contact")
            days = arguments.get("days")

            count = await asyncio.to_thread(
                sms_store.count_sms,
                contact=contact,
                days=days
            )

            time_desc = f"in the last {days} days" if days else "total"
            contact_desc = f" from {contact}" if contact else ""

            return {
                "success": True,
                "result": f"Found {count} SMS messages{contact_desc} ({time_desc})."
            }

        else:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing MCP tool {request.tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FastRTC Integration Endpoints

class ContextUpdateRequest(BaseModel):
    sms_messages: List[Dict[str, Any]] = []
    notifications: List[Dict[str, Any]] = []
    chat_history: List[Dict[str, Any]] = []

class VoiceQueryRequest(BaseModel):
    transcript: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded audio
    format: Optional[str] = "pcm"
    sample_rate: Optional[int] = 16000
    channels: Optional[int] = 1
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
    """Process voice query with audio data using STT and TTS"""
    if not VOICE_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice services not available")

    try:
        transcript = request.transcript

        # If audio data is provided, transcribe it first
        if request.audio_data and not transcript:
            logger.info("Processing audio data for STT...")

            # Decode base64 audio data
            try:
                audio_bytes = base64.b64decode(request.audio_data)

                # Convert to numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Use STT server to transcribe
                if stt_server:
                    transcript = await stt_server.transcribe_sync(audio_array)
                    logger.info(f"STT result: {transcript}")
                else:
                    transcript = "STT server not available"

            except Exception as e:
                logger.error(f"STT processing error: {e}")
                transcript = "Error processing audio"

        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript or audio data provided")

        # Generate AI response
        logger.info(f"Processing voice query: {transcript}")

        # Simple AI response (you can integrate with your actual AI model here)
        context_info = ""
        if request.context_data:
            notifications = request.context_data.get('notifications', [])
            messages = request.context_data.get('sms_messages', [])
            if notifications or messages:
                context_info = f" I can see you have {len(notifications)} notifications and {len(messages)} messages."

        ai_response = f"I heard you say: '{transcript}'. How can I help you?{context_info}"

        # Optional: Use TTS to generate audio response
        audio_response_base64 = None
        if tts_server:
            try:
                audio_data = await tts_server.synthesize(ai_response)
                if len(audio_data) > 0:
                    # Convert numpy array to bytes
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    audio_response_base64 = base64.b64encode(audio_bytes).decode()
                logger.info("TTS response generated")
            except Exception as e:
                logger.error(f"TTS error: {e}")

        return {
            "status": "success",
            "transcript": transcript,
            "response": ai_response,
            "audio_response": audio_response_base64,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fastrtc/status")
async def fastrtc_status():
    """Get voice streaming system status"""
    status = {
        "voice_services_available": VOICE_SERVICES_AVAILABLE,
        "stt_initialized": stt_server is not None,
        "tts_initialized": tts_server is not None,
        "timestamp": time.time()
    }

    if VOICE_SERVICES_AVAILABLE and stt_server and tts_server:
        status.update({
            "websocket_endpoint": "/ws",
            "transport": "websocket",
            "services": "faster-whisper + kokoro integrated",
            "status": "ready"
        })
    else:
        status.update({
            "status": "voice services not available"
        })

    return status

@app.get("/fastrtc/stream")
async def get_fastrtc_stream_info():
    """Get voice stream configuration for React Native client"""
    if not VOICE_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice services not available")

    try:
        # Return WebSocket connection details
        return {
            "stream_available": stt_server is not None and tts_server is not None,
            "transport": "websocket",
            "websocket_url": "/ws",
            "endpoints": {
                "context_update": "/fastrtc/context",
                "voice_query": "/fastrtc/voice-query",
                "status": "/fastrtc/status",
                "websocket": "/ws"
            },
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "encoding": "pcm"
            },
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Stream info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for WebRTC signaling only
@app.websocket("/ws")
async def websocket_signaling_endpoint(websocket: WebSocket):
    """WebSocket endpoint for WebRTC signaling (not for media streaming)"""
    await signaling_server.handle_connection(websocket)

# Data channel message processing endpoint
@app.post("/process-voice")
async def process_voice_message(data: dict = Body(...)):
    """Process voice messages received via WebRTC data channel"""
    try:
        message_type = data.get("type")

        if message_type == "audio":
            # Process audio data
            audio_base64 = data.get("audio")
            if not audio_base64:
                return {"error": "No audio data provided"}

            # Decode base64 audio
            import base64
            audio_bytes = base64.b64decode(audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            logger.info(f"Processing audio: {len(audio_array)} samples")

            # Transcribe using STT
            if stt_server:
                transcript = await stt_server.transcribe_audio_array(audio_array, sample_rate=16000)
            else:
                transcript = "Voice services not available"

            if not transcript or not transcript.strip():
                return {"type": "error", "error": "No speech detected"}

            logger.info(f"Transcript: {transcript}")

            # Get AI response
            ai_response = await call_agent_zero(transcript)

            # Generate TTS
            if tts_server:
                audio_output = await tts_server.synthesize(ai_response)
                # Convert to base64 for transmission
                audio_response_base64 = base64.b64encode(audio_output.tobytes()).decode('utf-8')
            else:
                audio_response_base64 = ""

            return {
                "type": "response",
                "transcript": transcript,
                "text": ai_response,
                "audio": audio_response_base64
            }

        elif message_type == "text":
            # Process text query directly
            text = data.get("text", "")
            ai_response = await call_agent_zero(text)

            # Generate TTS if requested
            if data.get("tts", False) and tts_server:
                audio_output = await tts_server.synthesize(ai_response)
                import base64
                audio_base64 = base64.b64encode(audio_output.tobytes()).decode('utf-8')
            else:
                audio_base64 = ""

            return {
                "type": "response",
                "text": ai_response,
                "audio": audio_base64
            }

        else:
            return {"error": f"Unknown message type: {message_type}"}

    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return {"type": "error", "error": str(e)}

# ================ MCP Tools for SMS Search ================

def get_sms_tools_schema() -> List[Dict]:
    """
    Get MCP tool schema definitions for SMS search capabilities.
    These tools can be provided to AgentZero for dynamic SMS context retrieval.
    """
    return [
        {
            "name": "search_sms",
            "description": "Semantically search through the user's SMS messages to find relevant conversations. Use this when the user asks about messages, texts, or specific information that might be in their SMS history.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant SMS messages"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of most relevant messages to return (default: 5)",
                        "default": 5
                    },
                    "time_window_days": {
                        "type": "integer",
                        "description": "Only search messages from the last N days (optional)",
                        "default": None
                    },
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_recent_sms",
            "description": "Get the most recent SMS messages in chronological order. Use this when the user asks about their latest messages or recent conversations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default: 5)",
                        "default": 5
                    },
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7
                    }
                },
                "required": []
            }
        },
        {
            "name": "count_sms",
            "description": "Count the total number of SMS messages in the database. Use this when the user asks how many messages they have.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    },
                    "days": {
                        "type": "integer",
                        "description": "Count only messages from the last N days (optional)",
                        "default": None
                    }
                },
                "required": []
            }
        }
    ]

async def execute_sms_tool(tool_name: str, tool_input: Dict) -> Dict:
    """
    Execute an SMS tool and return the result.

    Args:
        tool_name: Name of the tool to execute (search_sms, get_recent_sms, count_sms)
        tool_input: Input parameters for the tool

    Returns:
        Tool execution result
    """
    if not SMS_RAG_AVAILABLE or not sms_store:
        return {"error": "SMS RAG services not available"}

    try:
        if tool_name == "search_sms":
            results = await asyncio.to_thread(
                sms_store.search_sms,
                tool_input.get("query"),
                top_k=tool_input.get("top_k", 5),
                time_window_days=tool_input.get("time_window_days"),
                contact=tool_input.get("contact")
            )
            return {
                "results": results,
                "count": len(results),
                "tool": "search_sms"
            }

        elif tool_name == "get_recent_sms":
            results = await asyncio.to_thread(
                sms_store.get_recent_sms,
                limit=tool_input.get("limit", 5),
                contact=tool_input.get("contact"),
                days=tool_input.get("days", 7)
            )
            return {
                "results": results,
                "count": len(results),
                "tool": "get_recent_sms"
            }

        elif tool_name == "count_sms":
            count = await asyncio.to_thread(
                sms_store.count_sms,
                contact=tool_input.get("contact"),
                days=tool_input.get("days")
            )
            return {
                "count": count,
                "tool": "count_sms"
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.error(f"Error executing SMS tool {tool_name}: {e}")
        return {"error": str(e)}

def format_sms_results_for_context(results: List[Dict]) -> str:
    """
    Format SMS search results into a readable context string for the AI.

    Args:
        results: List of SMS message dictionaries

    Returns:
        Formatted context string
    """
    if not results:
        return "No matching SMS messages found."

    context_parts = []
    for i, msg in enumerate(results, 1):
        date_iso = msg.get('date_iso', '')
        address = msg.get('address', 'unknown')
        body = msg.get('body', '')
        similarity = msg.get('similarity', 0)

        context_parts.append(
            f"{i}. [{date_iso}] From {address} (relevance: {similarity:.2f}):\n   {body}"
        )

    return "\n\n".join(context_parts)

async def call_agent_zero(query_text: str, is_context_enhanced: bool = False, tools_enabled: bool = True) -> str:
    """
    Call AgentZero for AI response with enhanced context support and MCP tools.

    Args:
        query_text: The query or context to send
        is_context_enhanced: Whether the query includes additional context
        tools_enabled: Whether to provide SMS search tools to the agent

    Returns:
        AI response string
    """
    try:
        if is_context_enhanced:
            logger.info(f"🤖 Sending enhanced context query to AgentZero (length: {len(query_text)} chars)")
            logger.info(f"🤖 Query preview: {query_text[:200]}...")
        else:
            logger.info(f"🤖 Sending simple query to AgentZero: {query_text[:100]}...")

        # Get CSRF token
        csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
        csrf_response = requests.get(csrf_url, verify=False, timeout=10)
        csrf_response.raise_for_status()

        csrf_json = csrf_response.json()
        csrf_token = csrf_json.get("token")
        cookies = csrf_response.cookies

        if not csrf_token:
            raise ValueError("CSRF token not found")

        # Send query to AgentZero
        message_url = f"{AGENT_ZERO_URL}/message"
        headers = {
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json"
        }

        # Build payload with tools if enabled
        payload = {"text": query_text}

        # Add SMS tools information to the system context if available
        if tools_enabled and SMS_RAG_AVAILABLE and sms_store:
            tools_info = get_sms_tools_schema()
            sms_count = await asyncio.to_thread(sms_store.count_sms)

            # Get the base URL of our server
            server_url = "https://zswok4sc8c44w804kw8gss8g.uptopoint.net"

            # Append tool availability information with MCP endpoints
            tools_context = f"""

--- Available MCP Tools ---
You have access to the user's SMS message database ({sms_count} messages indexed).

To use SMS tools, you can call these endpoints:

1. **search_sms** - Search SMS semantically
   Endpoint: POST {server_url}/mcp/call-tool
   Payload: {{"tool_name": "search_sms", "arguments": {{"query": "your search query", "top_k": 10}}}}

2. **get_recent_sms** - Get recent SMS messages
   Endpoint: POST {server_url}/mcp/call-tool
   Payload: {{"tool_name": "get_recent_sms", "arguments": {{"limit": 10, "days": 7}}}}

3. **count_sms** - Count total SMS
   Endpoint: POST {server_url}/mcp/call-tool
   Payload: {{"tool_name": "count_sms", "arguments": {{"days": 30}}}}

Tool List: GET {server_url}/mcp/tools

IMPORTANT: When the user asks about SMS, messages, or money transactions, USE THESE TOOLS by making HTTP POST requests to retrieve the actual data. Don't just mention that you need the information - actively fetch it using the endpoints above."""

            payload["text"] = query_text + tools_context
            logger.info(f"🔧 Added MCP SMS tools context ({sms_count} messages available)")

        response = requests.post(
            message_url,
            headers=headers,
            cookies=cookies,
            json=payload,
            verify=False,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        ai_response = result.get("message", "I'm sorry, I couldn't process that request.")

        if is_context_enhanced:
            logger.info(f"✅ AgentZero context-enhanced response: {ai_response[:100]}...")
        else:
            logger.info(f"✅ AgentZero simple response: {ai_response[:100]}...")

        return ai_response

    except requests.exceptions.HTTPError as e:
        logger.error(f"AgentZero HTTP error: {e}")

        # Check for rate limit errors (500 Internal Server Error often indicates rate limiting)
        if hasattr(e, 'response') and e.response.status_code == 500:
            logger.warning("⚠️ Detected potential rate limiting from Gemini API")
            return "I'm temporarily busy processing other requests. Please try again in a moment."
        elif hasattr(e, 'response') and e.response.status_code == 429:
            logger.warning("⚠️ Detected rate limiting (429) from Gemini API")
            return "I'm receiving too many requests right now. Please wait a moment and try again."
        else:
            return "I'm sorry, there was an error processing your request. Please try again."

    except requests.exceptions.Timeout:
        logger.error("AgentZero timeout error")
        return "The request took too long to process. Please try a shorter message."

    except Exception as e:
        logger.error(f"AgentZero error: {e}")
        return "I'm sorry, there was an error processing your request."

# Global start time for uptime tracking
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Proxy Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)  # Use port 8001 to avoid conflicts
