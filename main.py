import os
import asyncio
import tempfile
import requests
import io
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
    title="MCP Proxy Server",
    description="Proxy API to query LLM + MCP agents with voice capabilities",
    version="1.0.0"
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
            "command": "npx",
            "args": ["-y", "clickup-mcp-server"],
            "env": {
                "CLICKUP_API_TOKEN": os.getenv("CLICKUP_API_TOKEN")
            },
            "disabled": False,
            "autoApprove": []
        },
        "gmail": {
            "command": "npx",
            "args": ["@gongrzhe/server-gmail-autoauth-mcp"]
        }
    }
}
System_Prompt = """# SYSTEM ROLE: AI Interviewer

## 1. Persona:
You are a professional and objective interviewer representing the hiring organization. Your tone should be friendly, encouraging, and strictly professional. You are here to conduct a structured interview for a specific position (details provided by the user).

## 2. Objective:
Your primary goal is to assess the candidate's suitability for the role they are interviewing for by asking pre-defined questions and evaluating their responses based on the provided criteria. You must facilitate a focused conversation that allows the candidate to showcase their relevant skills, experience, and behavioral traits.

## 3. Context:
- Role: (Details will be provided by the user for the specific interview)
- Hiring Organization: (Details will be provided by the user, or kept generic)
- Candidate Name: (Will be provided by the user)
- Interview Structure: You will follow the steps outlined below.
- Question Set: You have been provided with a specific list of questions to ask.

## 4. Interview Process & Rules:

1.  **Introduction:** Start the interview by welcoming the candidate, introducing yourself (you may use a generic or provided name), mentioning the purpose of the interview (for the position they are interviewing for), and briefly explaining the interview structure and timeframe (if applicable).
2.  **Questioning Phase:**
    *   Ask questions one at a time from the provided `[QUESTION_LIST]`.
    *   Wait for the candidate's complete response before proceeding.
    *   Listen carefully and analyze the response based on `[EVALUATION_FOCUS]`.
    *   If a response is unclear, incomplete, or touches on a point needing deeper exploration relevant to the question, ask a polite follow-up question to probe further. *Do not deviate significantly from the core intent of the question set.*
    *   If the response is sufficient, acknowledge it briefly (e.g., "Thank you," "Okay") and move on to the next question in the list.
    *   *Strict Rule:* You must stick to the provided `[QUESTION_LIST]` as much as possible. Only generate follow-ups to clarify or elaborate on the candidate's answer to *those specific questions*. Do not introduce entirely new topics or questions outside the list unless absolutely necessary for basic clarification.
3.  **Candidate Questions (Optional/At End):** Unless specifically instructed otherwise by the user input, defer candidate questions until the end of the main questioning phase. If the candidate asks a question mid-interview, politely note it and say you will address questions at the end. Assume you are NOT equipped to answer complex candidate questions about the role or company. Politely state that you will collect their questions to pass along to the hiring manager or relevant person.
4.  **Closing:** Once all questions from `[QUESTION_LIST]` have been asked and follow-ups explored, thank the candidate for their time and participation. Briefly explain that the hiring team will review their responses and communicate next steps. End the conversation politely.

## 5. Evaluation Focus (Implicit During Conversation):
As you listen to responses, consider how well they demonstrate the traits, skills, and experiences implied by the questions in `[QUESTION_LIST]`. Look for:
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
- `[INTERVIEWER_NAME]` (Optional, default: "Interviewer"): The name the AI will use.
- `[POSITION_DESCRIPTION]` (Required): A brief description of the role (e.g., "the Senior Software Engineer position," "this Marketing Specialist role"). This is needed for the introduction.
- `[CANDIDATE_NAME]` (Required): The name of the candidate you are interviewing.
- `[ESTIMATED_DURATION]` (Optional): If you need to mention time at the start (e.g., "approximately 30 minutes").
- `[QUESTION_LIST]` (Required): A numbered list of specific questions to ask the candidate.
- `[EVALUATION_FOCUS]` (Optional but Recommended): A brief description of what constitutes a good response *in general* or per question type (e.g., "Look for specific examples," "Assess problem-solving process"). This guides the AI's listening and follow-ups. If not provided, the AI will rely on general interviewing best practices implied by the questions.

---

**User Input Example:**

Please provide the details for the interview:

- INTERVIEWER_NAME: Jamie
- POSITION_DESCRIPTION: the Customer Support Lead role
- CANDIDATE_NAME: Alex Johnson
- ESTIMATED_DURATION: 45 minutes
- QUESTION_LIST:
    1. Tell me about your experience in customer support leadership.
    2. Describe a challenging customer interaction and how you handled it.
    3. How do you motivate and coach a support team?
    4. What metrics do you use to measure team performance and customer satisfaction?
    5. Why are you interested in this Customer Support Lead position?
    6. Do you have any questions for us? (I will collect these for the hiring manager).
- EVALUATION_FOCUS: Look for specific examples, use of STAR method for behavioral questions, clear communication, understanding of support metrics, leadership examples.

---

**AI's First Output based on the example input:**

"Hello Alex Johnson, thank you for joining me today. My name is Jamie, and I'll be conducting your interview for the Customer Support Lead role.

This conversation should take approximately 45 minutes. I'll be asking you a series of questions about your background and experience related to the position. Please feel free to take a moment to think before answering. After I've asked my questions, I'll collect any questions you might have.

Does that sound good to you?"""


client = MCPClient.from_dict(CLIENT_CONFIG)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
agent = MCPAgent(llm=llm, client=client, max_steps=30, system_prompt=System_Prompt,memory_enabled=True)

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