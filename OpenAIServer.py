import os
import asyncio
import tempfile
import requests
import io
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from mcp_use import MCPAgent, MCPClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="OpenAI-Compatible MCP Proxy",
    description="Proxy API to query LLM + MCP agents with Groq STT/TTS via OpenAI endpoints",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP client configuration
CLIENT_CONFIG = {
    "mcpServers": {
        "clickup": {
            "command": "npx",
            "args": ["-y", "clickup-mcp-server"],
            "env": {"CLICKUP_API_TOKEN": os.getenv("CLICKUP_API_TOKEN")},
            "disabled": False,
            "autoApprove": []
        },
        "gmail": {"command": "npx", "args": ["@gongrzhe/server-gmail-autoauth-mcp"]}
    }
}

# System prompt
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "# SYSTEM ROLE: AI Interviewer... (your default here)")

# Initialize MCP agent\ nclient = MCPClient.from_dict(CLIENT_CONFIG)
client = MCPClient.from_dict(CLIENT_CONFIG)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
agent = MCPAgent(llm=llm, client=client, max_steps=30, system_prompt=SYSTEM_PROMPT, memory_enabled=True)

# Supported models list
SUPPORTED_MODELS = [
    "gemini-2.0-flash",
    "whisper-large-v3-turbo",
    "playai-tts"
]

# Pydantic schemas
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mcp"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]

class AudioSynthesisRequest(BaseModel):
    model: str = Field(default="playai-tts")
    voice: str = Field(default="Fritz-PlayAI")
    input: str
    response_format: str = Field(default="wav")

# Models endpoints
@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(data=[ModelInfo(id=m) for m in SUPPORTED_MODELS])

@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelInfo(id=model_id)

# Chat endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    system_msg = next((m.content for m in request.messages if m.role == "system"), None)
    user_content = "\n".join(m.content for m in request.messages if m.role == "user")
    prompt = f"{system_msg}\n{user_content}" if system_msg else user_content
    try:
        result = await agent.run(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    choice = ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=result), finish_reason="stop")
    return ChatCompletionResponse(id="mcp-"+os.urandom(8).hex(), created=int(asyncio.get_event_loop().time()), model=request.model, choices=[choice])

# Audio transcription (STT)
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form("whisper-large-v3-turbo")):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
    files = {"file": (file.filename, open(tmp_path, "rb"), file.content_type)}
    data = {"model": model, "response_format": "json", "language": "en", "temperature": 0.0}
    resp = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files, data=data)
    os.unlink(tmp_path)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()

# Audio synthesis (TTS)
@app.post("/v1/audio/speech")
async def openai_speech_synthesis(request: AudioSynthesisRequest = Body(...)):
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    payload = request.dict()
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
    resp = requests.post("https://api.groq.com/openai/v1/audio/speech", headers=headers, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    audio_bytes = io.BytesIO(resp.content)
    return StreamingResponse(audio_bytes, media_type="audio/wav", headers={"Content-Disposition":"attachment; filename=speech.wav"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
