#!/usr/bin/env python3
"""
FastRTC Voice Handler - Integrates faster-whisper STT and Kokoro TTS with FastRTC
Provides real-time voice processing with automatic voice activity detection
"""

import asyncio
import logging
import numpy as np
import tempfile
import soundfile as sf
import io
import requests
from typing import AsyncGenerator, Tuple, Optional
from fastrtc import Stream, ReplyOnPause
from faster_whisper_stt_server import FasterWhisperSTTServer
from kokoro_tts_server import KokoroTTSServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastRTCVoiceHandler:
    def __init__(self, agent_zero_url: str = "https://ao.uptopoint.net"):
        """Initialize FastRTC voice handler with STT and TTS servers"""
        self.agent_zero_url = agent_zero_url

        # Initialize STT and TTS servers
        logger.info("Initializing faster-whisper STT server...")
        self.stt_server = FasterWhisperSTTServer(
            model_name="tiny",  # Fast model for real-time
            device="auto",
            compute_type="int8"
        )

        logger.info("Initializing Kokoro TTS server...")
        self.tts_server = KokoroTTSServer(
            device="auto",
            sample_rate=24000
        )

        # Load models
        asyncio.create_task(self.initialize_models())

    async def initialize_models(self):
        """Initialize STT and TTS models asynchronously"""
        try:
            logger.info("Loading faster-whisper model...")
            await self.stt_server.initialize()

            logger.info("Loading Kokoro TTS model...")
            await self.tts_server.initialize()

            logger.info("✅ All models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise

    async def call_agent_zero(self, transcript: str) -> str:
        """Call AgentZero for AI response"""
        try:
            # Get CSRF token
            csrf_url = f"{self.agent_zero_url}/csrf_token"
            csrf_response = requests.get(csrf_url, verify=False, timeout=10)
            csrf_response.raise_for_status()

            csrf_json = csrf_response.json()
            csrf_token = csrf_json.get("token")
            cookies = csrf_response.cookies

            if not csrf_token:
                raise ValueError("CSRF token not found")

            # Send query to AgentZero
            query_url = f"{self.agent_zero_url}/query"
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }

            payload = {"query": transcript}
            response = requests.post(
                query_url,
                headers=headers,
                cookies=cookies,
                json=payload,
                verify=False,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            return result.get("result", "I'm sorry, I couldn't process that request.")

        except Exception as e:
            logger.error(f"AgentZero error: {e}")
            return "I'm sorry, there was an error processing your request."

    async def voice_handler(self, audio: Tuple[int, np.ndarray]) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Main voice processing handler for FastRTC

        Args:
            audio: Tuple of (sample_rate, audio_array) from user's microphone

        Yields:
            Tuple of (sample_rate, audio_array) for TTS response
        """
        try:
            sample_rate, audio_array = audio
            logger.info(f"🎤 Received audio: {audio_array.shape}, sample_rate: {sample_rate}")

            # Convert audio to the format expected by faster-whisper
            if audio_array.ndim > 1:
                # Convert stereo to mono if needed
                audio_array = np.mean(audio_array, axis=0)

            # Ensure audio is float32 and normalized
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))

            # Transcribe using faster-whisper
            logger.info("🔍 Transcribing audio with faster-whisper...")
            transcript = await self.stt_server.transcribe_audio_array(
                audio_array,
                sample_rate=sample_rate
            )

            if not transcript or not transcript.strip():
                logger.info("🔇 No speech detected, not responding")
                return

            logger.info(f"📝 Transcription: {transcript}")

            # Get AI response from AgentZero
            logger.info("🤖 Getting AI response...")
            ai_response = await self.call_agent_zero(transcript)
            logger.info(f"💬 AI Response: {ai_response[:100]}...")

            # Generate TTS using Kokoro with streaming
            logger.info("🔊 Generating TTS with Kokoro...")
            async for audio_chunk in self.tts_server.synthesize_streaming(ai_response):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Convert to numpy array if needed
                    if hasattr(audio_chunk, 'numpy'):
                        audio_chunk = audio_chunk.numpy()
                    elif not isinstance(audio_chunk, np.ndarray):
                        audio_chunk = np.array(audio_chunk)

                    # Ensure proper shape for FastRTC (1, num_samples)
                    if audio_chunk.ndim == 1:
                        audio_chunk = audio_chunk.reshape(1, -1)

                    # Ensure float32 format
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32)

                    logger.debug(f"🎵 Yielding TTS chunk: {audio_chunk.shape}")
                    yield (24000, audio_chunk)  # Kokoro uses 24kHz

            logger.info("✅ Voice processing completed")

        except Exception as e:
            logger.error(f"❌ Voice handler error: {e}", exc_info=True)
            # Generate error response
            error_text = "I'm sorry, there was an error processing your request."
            try:
                async for audio_chunk in self.tts_server.synthesize_streaming(error_text):
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        if hasattr(audio_chunk, 'numpy'):
                            audio_chunk = audio_chunk.numpy()
                        elif not isinstance(audio_chunk, np.ndarray):
                            audio_chunk = np.array(audio_chunk)

                        if audio_chunk.ndim == 1:
                            audio_chunk = audio_chunk.reshape(1, -1)

                        if audio_chunk.dtype != np.float32:
                            audio_chunk = audio_chunk.astype(np.float32)

                        yield (24000, audio_chunk)
            except Exception as tts_error:
                logger.error(f"❌ Error TTS failed: {tts_error}")

def create_fastrtc_stream(agent_zero_url: str = "https://ao.uptopoint.net") -> Stream:
    """Create and configure FastRTC stream with voice handler using WebSocket mode"""

    # Create voice handler instance
    voice_handler = FastRTCVoiceHandler(agent_zero_url)

    # Create FastRTC stream with ReplyOnPause for automatic voice activity detection
    stream = Stream(
        handler=ReplyOnPause(voice_handler.voice_handler),
        modality="audio",
        mode="send-receive"
    )

    logger.info("✅ FastRTC stream created successfully")
    return stream