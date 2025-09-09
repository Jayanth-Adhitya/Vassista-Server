#!/usr/bin/env python3
"""
FastRTC Voice Agent - Integrates STT, TTS, and AgentZero for real-time voice interaction
Provides ultra-low latency voice conversations using FastRTC framework
"""

import asyncio
import logging
import numpy as np
import requests
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any
from fastrtc import Stream, ReplyOnPause, get_stt_model, get_tts_model
from contextlib import asynccontextmanager

# Import our local servers
from kimi_stt_server import get_stt_server, process_audio_for_fastrtc
from kokoro_tts_server import get_tts_server, synthesize_for_fastrtc, synthesize_base64_for_fastrtc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AgentZero configuration
AGENT_ZERO_URL = "https://ao.uptopoint.net"

class FastRTCVoiceAgent:
    def __init__(self):
        self.stt_server = None
        self.tts_server = None
        self.context_cache = {}
        
    async def initialize(self):
        """Initialize STT and TTS servers"""
        try:
            logger.info("Initializing FastRTC Voice Agent...")
            
            # Initialize local servers
            self.stt_server = await get_stt_server()
            self.tts_server = await get_tts_server()
            
            logger.info("FastRTC Voice Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FastRTC Voice Agent: {e}")
            return False
    
    async def get_agent_zero_response(self, context: str) -> str:
        """Get response from AgentZero"""
        try:
            # Fetch CSRF token and cookies
            csrf_url = f"{AGENT_ZERO_URL}/csrf_token"
            csrf_response = requests.get(csrf_url, verify=False)
            csrf_response.raise_for_status()
            
            csrf_json = csrf_response.json()
            csrf_token = csrf_json.get("token")
            cookies = csrf_response.cookies
            
            if not csrf_token:
                raise ValueError(f"CSRF token not found: {csrf_json}")

            # Send message to AgentZero
            message_url = f"{AGENT_ZERO_URL}/message"
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }
            
            payload = {"text": context}
            
            agent_response = requests.post(
                message_url, 
                headers=headers, 
                cookies=cookies, 
                json=payload, 
                verify=False,
                timeout=30  # 30 second timeout for real-time
            )
            
            agent_response.raise_for_status()
            agent_result_json = agent_response.json()
            result_text = agent_result_json.get("message", str(agent_result_json))
            
            return result_text
            
        except requests.exceptions.Timeout:
            logger.error("AgentZero timeout")
            return "I'm thinking... please try again."
        except Exception as e:
            logger.error(f"AgentZero error: {e}")
            return "I encountered an error. Please try again."
    
    async def process_voice_input(self, transcript: str) -> str:
        """Process voice input and get response"""
        try:
            if not transcript.strip():
                return ""
            
            logger.info(f"Processing voice input: {transcript}")
            
            # For now, use the transcript directly as context
            # In production, you'd add SMS, notifications, chat history here
            context = f"User Query: {transcript}"
            
            # Get response from AgentZero
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.get_agent_zero_response, context
            )
            
            logger.info(f"AgentZero response: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return "Sorry, I couldn't process that. Please try again."

# Create global agent instance
voice_agent = FastRTCVoiceAgent()

async def voice_conversation_handler(audio_stream):
    """
    Main conversation handler for FastRTC
    This function processes incoming audio and returns responses
    """
    try:
        # Ensure agent is initialized
        if voice_agent.stt_server is None or voice_agent.tts_server is None:
            await voice_agent.initialize()
        
        # Convert audio stream to numpy array for processing
        # This is a placeholder - actual implementation depends on FastRTC audio format
        audio_data = np.array(audio_stream, dtype=np.float32)
        
        # Transcribe audio using Kimi-Audio
        transcript = await process_audio_for_fastrtc(audio_data)
        
        if transcript:
            logger.info(f"Transcribed: {transcript}")
            
            # Get response from AgentZero
            response = await voice_agent.process_voice_input(transcript)
            
            if response:
                logger.info(f"Response: {response[:100]}...")
                
                # Synthesize response using Kokoro
                audio_response = await synthesize_for_fastrtc(response)
                
                return audio_response
        
        return np.array([], dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Conversation handler error: {e}")
        return np.array([], dtype=np.float32)

def create_fastrtc_stream():
    """Create and configure FastRTC stream"""
    try:
        # Create a simple echo function wrapped with ReplyOnPause
        async def process_audio_chunk(audio_chunk):
            """Process individual audio chunks"""
            try:
                # This would be the main processing logic
                response_audio = await voice_conversation_handler(audio_chunk)
                return response_audio
            except Exception as e:
                logger.error(f"Audio chunk processing error: {e}")
                return np.array([], dtype=np.float32)
        
        # Wrap with ReplyOnPause for automatic turn detection
        reply_on_pause = ReplyOnPause(process_audio_chunk)
        
        # Create FastRTC stream
        stream = Stream(
            handler=reply_on_pause,
            # Additional configuration can be added here
        )
        
        return stream
        
    except Exception as e:
        logger.error(f"FastRTC stream creation error: {e}")
        return None

# Alternative implementation using FastRTC's built-in models as fallback
def create_fastrtc_stream_with_builtin():
    """Create FastRTC stream using built-in models (fallback)"""
    try:
        # Use FastRTC's built-in STT and TTS models
        stt_model = get_stt_model(model="whisper")  # or moonshine if available
        tts_model = get_tts_model(model="kokoro")
        
        async def conversation_logic(transcript: str) -> str:
            """Main conversation logic"""
            try:
                if not transcript.strip():
                    return ""
                
                logger.info(f"Processing: {transcript}")
                
                # Get response from AgentZero
                response = await voice_agent.process_voice_input(transcript)
                return response
                
            except Exception as e:
                logger.error(f"Conversation logic error: {e}")
                return "I encountered an error. Please try again."
        
        # Wrap with ReplyOnPause
        reply_on_pause = ReplyOnPause(conversation_logic)
        
        # Create stream
        stream = Stream(
            handler=reply_on_pause,
            stt_model=stt_model,
            tts_model=tts_model
        )
        
        return stream
        
    except Exception as e:
        logger.error(f"Built-in FastRTC stream creation error: {e}")
        return None

# Context management for mobile app integration
class MobileContextManager:
    """Manages context from mobile app (SMS, notifications, chat history)"""
    
    def __init__(self):
        self.context_data = {}
    
    async def update_context(self, context_data: Dict[str, Any]):
        """Update context data from mobile app"""
        self.context_data = context_data
        logger.info(f"Updated mobile context: {len(context_data)} items")
    
    def build_full_context(self, user_query: str) -> str:
        """Build full context string for AgentZero"""
        context_parts = [f"User Query: {user_query}"]
        
        # Add SMS messages
        if "sms_messages" in self.context_data:
            context_parts.append("\n\n--- Recent Phone Messages ---")
            for msg in self.context_data["sms_messages"][:10]:
                context_parts.append(f"From: {msg.get('address', 'Unknown')}\n{msg.get('body', '')}\n")
        
        # Add notifications
        if "notifications" in self.context_data:
            context_parts.append("\n\n--- Recent Notifications ---")
            for notif in self.context_data["notifications"][:10]:
                context_parts.append(f"App: {notif.get('packageName', 'Unknown')}\n"
                                    f"Title: {notif.get('title', '')}\n"
                                    f"Body: {notif.get('body', '')}\n")
        
        # Add chat history
        if "chat_history" in self.context_data:
            context_parts.append("\n\n--- Recent Chat History ---")
            for msg in self.context_data["chat_history"][-10:]:
                context_parts.append(f"{msg.get('sender', 'unknown')}: {msg.get('text', '')}")
        
        return '\n'.join(context_parts)

# Global context manager
mobile_context = MobileContextManager()

# Main entry point functions
async def initialize_agent():
    """Initialize the FastRTC voice agent"""
    return await voice_agent.initialize()

def get_fastrtc_stream():
    """Get configured FastRTC stream for the application"""
    # Try custom implementation first, fallback to built-in models
    stream = create_fastrtc_stream()
    if stream is None:
        logger.warning("Custom FastRTC stream failed, using built-in models")
        stream = create_fastrtc_stream_with_builtin()
    
    return stream

async def update_mobile_context(context_data: Dict[str, Any]):
    """Update mobile context data"""
    await mobile_context.update_context(context_data)

if __name__ == "__main__":
    async def main():
        # Test the voice agent
        success = await initialize_agent()
        
        if success:
            logger.info("FastRTC Voice Agent ready!")
            
            # Test conversation
            test_input = "Hello, how are you today?"
            response = await voice_agent.process_voice_input(test_input)
            logger.info(f"Test response: {response}")
            
            # Test stream creation
            stream = get_fastrtc_stream()
            if stream:
                logger.info("FastRTC stream created successfully!")
            else:
                logger.error("Failed to create FastRTC stream")
                
        else:
            logger.error("Failed to initialize voice agent")
    
    asyncio.run(main())