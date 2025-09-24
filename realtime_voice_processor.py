"""
Real-time Voice Processor for WebRTC Streaming
Handles continuous STT and streaming TTS for true real-time conversation
"""

import asyncio
import numpy as np
import logging
from typing import AsyncGenerator, Optional
import json
from collections import deque
import time

logger = logging.getLogger(__name__)

class RealtimeVoiceProcessor:
    """Process WebRTC audio streams in real-time with streaming STT and TTS"""

    def __init__(self, stt_server, tts_server, agent_callback):
        self.stt_server = stt_server
        self.tts_server = tts_server
        self.agent_callback = agent_callback  # Function to get AI responses

        # Audio buffers
        self.audio_buffer = deque(maxlen=100)  # Rolling buffer for audio chunks
        self.sample_rate = 16000

        # STT state
        self.is_listening = False
        self.speech_buffer = []
        self.last_speech_time = 0
        self.silence_threshold = 1.5  # Seconds of silence before processing

        # TTS state
        self.is_speaking = False
        self.tts_queue = asyncio.Queue()

        # Conversation state
        self.current_transcript = ""
        self.conversation_context = []

        logger.info("Real-time voice processor initialized")

    async def process_audio_stream(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process incoming audio chunk and return TTS audio if available
        Real-time pipeline: Audio → STT → AI → TTS → Audio
        """

        # Add to buffer for STT processing
        self.audio_buffer.append(audio_chunk)

        # Process STT in parallel
        asyncio.create_task(self._process_stt())

        # Check if we have TTS audio to return
        if not self.tts_queue.empty():
            try:
                tts_audio = await asyncio.wait_for(self.tts_queue.get(), timeout=0.01)
                return tts_audio
            except asyncio.TimeoutError:
                pass

        return None

    async def _process_stt(self):
        """Continuous STT processing with VAD"""
        if not self.is_listening or len(self.audio_buffer) < 5:
            return

        # Combine recent audio chunks
        audio_data = np.concatenate(list(self.audio_buffer)[-10:])  # Last ~0.5 seconds

        try:
            # Use faster-whisper's VAD to detect speech
            segments, info = await self._transcribe_chunk(audio_data)

            if segments:
                # Speech detected
                self.last_speech_time = time.time()

                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        self.current_transcript += " " + text
                        logger.info(f"Partial transcript: {text}")

                        # Emit partial transcript
                        await self._emit_transcript(text, is_final=False)

            # Check for end of utterance (silence threshold)
            elif time.time() - self.last_speech_time > self.silence_threshold:
                if self.current_transcript.strip():
                    # Process complete utterance
                    await self._process_complete_utterance()

        except Exception as e:
            logger.error(f"STT processing error: {e}")

    async def _transcribe_chunk(self, audio_chunk: np.ndarray):
        """Transcribe a single audio chunk"""
        # Use faster-whisper with VAD
        return await asyncio.to_thread(
            self.stt_server.model.transcribe,
            audio_chunk,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=10
            )
        )

    async def _process_complete_utterance(self):
        """Process a complete user utterance"""
        transcript = self.current_transcript.strip()
        if not transcript:
            return

        logger.info(f"Complete utterance: {transcript}")

        # Reset current transcript
        self.current_transcript = ""

        # Emit final transcript
        await self._emit_transcript(transcript, is_final=True)

        # Get AI response (non-blocking)
        asyncio.create_task(self._generate_response(transcript))

    async def _generate_response(self, transcript: str):
        """Generate AI response and queue TTS audio"""
        try:
            # Get AI response
            response = await self.agent_callback(transcript)
            logger.info(f"AI response: {response[:100]}...")

            # Generate TTS audio with streaming
            await self._stream_tts(response)

        except Exception as e:
            logger.error(f"Response generation error: {e}")

    async def _stream_tts(self, text: str):
        """Stream TTS audio generation"""
        self.is_speaking = True

        try:
            # Use Kokoro's streaming synthesis
            async for audio_chunk in self.tts_server.synthesize_streaming(text):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Queue audio for playback
                    await self.tts_queue.put(audio_chunk)

                    # Small delay to prevent buffer overflow
                    await asyncio.sleep(0.01)

        finally:
            self.is_speaking = False

    async def _emit_transcript(self, text: str, is_final: bool):
        """Emit transcript event (to be connected to WebRTC data channel)"""
        # This will be connected to the WebRTC data channel
        event = {
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "timestamp": time.time()
        }
        # To be implemented: Send via data channel
        logger.debug(f"Transcript event: {event}")

    async def start_listening(self):
        """Start listening for voice input"""
        self.is_listening = True
        self.current_transcript = ""
        self.last_speech_time = time.time()
        logger.info("Started listening")

    async def stop_listening(self):
        """Stop listening for voice input"""
        self.is_listening = False

        # Process any remaining transcript
        if self.current_transcript.strip():
            await self._process_complete_utterance()

        logger.info("Stopped listening")

    def is_active(self):
        """Check if processor is active"""
        return self.is_listening or self.is_speaking


class WebRTCAudioHandler:
    """Handle WebRTC audio tracks with real-time processing"""

    def __init__(self, voice_processor: RealtimeVoiceProcessor):
        self.voice_processor = voice_processor
        self.input_track = None
        self.output_track = None
        self.processing_task = None

    async def set_input_track(self, track):
        """Set the input audio track from WebRTC"""
        self.input_track = track

        # Start processing audio
        if self.processing_task:
            self.processing_task.cancel()

        self.processing_task = asyncio.create_task(self._process_input_audio())

    async def _process_input_audio(self):
        """Continuously process input audio"""
        while self.input_track:
            try:
                # Get audio frame from WebRTC track
                frame = await self.input_track.recv()

                # Convert to numpy array
                audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16).astype(np.float32) / 32768.0

                # Process through voice processor
                output_audio = await self.voice_processor.process_audio_stream(audio_data)

                # Send output audio if available
                if output_audio is not None and self.output_track:
                    # Convert back to WebRTC frame format
                    # This would be sent to the output track
                    pass

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)

    def stop(self):
        """Stop audio processing"""
        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None