#!/usr/bin/env python3
"""
Faster Whisper STT Server - High-performance CPU-optimized speech-to-text
Uses faster-whisper with CTranslate2 for optimal CPU performance
"""

import asyncio
import logging
import numpy as np
from typing import AsyncGenerator
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FasterWhisperSTTServer:
    def __init__(self, model_name="tiny", device="auto", compute_type="int8"):
        self.model_name = model_name
        self.compute_type = compute_type  # int8 for CPU optimization
        self.device = self._get_device(device)
        self.model = None
        self.sample_rate = 16000  # Standard sample rate for speech recognition
        self.executor = ThreadPoolExecutor(max_workers=2)  # For async processing
        
    def _get_device(self, device):
        if device == "auto":
            # faster-whisper handles CPU optimization automatically
            device = "cpu"
            logger.info("Using CPU with faster-whisper optimizations")
        return device
    
    async def initialize(self):
        """Initialize the faster-whisper model"""
        try:
            logger.info(f"Loading faster-whisper model: {self.model_name}")
            logger.info(f"Device: {self.device}, Compute type: {self.compute_type}")
            
            # Load faster-whisper model with CPU optimizations
            self.model = WhisperModel(
                self.model_name, 
                device=self.device,
                compute_type=self.compute_type,  # int8 quantization for CPU
                cpu_threads=0,  # Use all available CPU threads
                num_workers=1   # Single worker for better memory usage
            )
            
            logger.info("Faster-whisper model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper model: {e}")
            return False
    
    def preprocess_audio(self, audio_data: np.ndarray, original_sr: int = None) -> np.ndarray:
        """Preprocess audio data for faster-whisper"""
        try:
            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure single channel
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize audio to [-1, 1] range
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
            # Resample if necessary (faster-whisper expects 16kHz)
            if original_sr and original_sr != self.sample_rate:
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=original_sr, 
                    target_sr=self.sample_rate
                )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription for use in thread pool"""
        try:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                beam_size=1,  # Faster inference with beam_size=1
                best_of=1,    # Faster inference
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,  # Faster for short clips
                word_timestamps=False  # Disable for speed
            )
            
            # Combine all segments into single transcription
            transcription = ""
            for segment in segments:
                transcription += segment.text.strip() + " "
            
            return transcription.strip()
                
        except Exception as e:
            logger.error(f"Sync transcription error: {e}")
            return ""
    
    async def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio array directly (for FastRTC integration)

        Args:
            audio_array: numpy array of audio data
            sample_rate: sample rate of the audio

        Returns:
            str: transcribed text
        """
        try:
            logger.info(f"Transcribing audio array: shape={audio_array.shape}, sr={sample_rate}")

            # Ensure we have a 1D array
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=0)

            # Ensure proper format for faster-whisper
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Resample if needed
            if sample_rate != self.sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_array
            )

            logger.info(f"Transcription result: {transcription}")
            return transcription

        except Exception as e:
            logger.error(f"Audio array transcription error: {e}")
            return ""

    async def transcribe_chunk(self, audio_chunk: np.ndarray, original_sr: int = None) -> str:
        """Transcribe a single audio chunk using faster-whisper (async)"""
        try:
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_chunk, original_sr)
            if len(audio_data) == 0:
                return ""
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                self.executor, 
                self._transcribe_sync, 
                audio_data
            )
            
            return transcription
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """Process streaming audio and yield transcriptions"""
        buffer = []
        chunk_duration = 3.0  # Process every 3 seconds for better accuracy
        chunk_size = int(self.sample_rate * chunk_duration)
        
        async for audio_bytes in audio_stream:
            try:
                # Convert bytes to numpy array
                # Assuming 16-bit PCM audio
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                buffer.extend(audio_data)
                
                # Process when we have enough data
                if len(buffer) >= chunk_size:
                    chunk = np.array(buffer[:chunk_size])
                    buffer = buffer[chunk_size//2:]  # Overlap for better continuity
                    
                    transcription = await self.transcribe_chunk(chunk)
                    if transcription:
                        yield transcription
                        
            except Exception as e:
                logger.error(f"Streaming transcription error: {e}")
                continue
    
    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio from file"""
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(file_path)
            
            # Transcribe
            transcription = await self.transcribe_chunk(audio_data, sample_rate)
            return transcription
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

# Global STT server instance
stt_server = None

async def get_stt_server():
    """Get or create the global STT server instance"""
    global stt_server
    if stt_server is None:
        stt_server = FasterWhisperSTTServer()
        success = await stt_server.initialize()
        if not success:
            stt_server = None
            raise RuntimeError("Failed to initialize Faster Whisper STT server")
    return stt_server

# Utility function for FastRTC integration (same interface as kimi_stt_server)
async def process_audio_for_fastrtc(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Process audio data for FastRTC integration"""
    try:
        server = await get_stt_server()
        return await server.transcribe_chunk(audio_data, sample_rate)
    except Exception as e:
        logger.error(f"FastRTC audio processing error: {e}")
        return ""

if __name__ == "__main__":
    async def main():
        # Test the STT server
        server = FasterWhisperSTTServer()
        success = await server.initialize()
        
        if success:
            logger.info("Faster Whisper STT Server initialized successfully!")
            
            # Test with a dummy audio chunk (silence)
            dummy_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of silence
            result = await server.transcribe_chunk(dummy_audio)
            logger.info(f"Test transcription result: '{result}'")
            
            # Test with some noise (should return empty or minimal result)
            noise_audio = np.random.randn(16000 * 1).astype(np.float32) * 0.01
            result2 = await server.transcribe_chunk(noise_audio)
            logger.info(f"Noise transcription result: '{result2}'")
            
            server.cleanup()
        else:
            logger.error("Failed to initialize STT server")
    
    asyncio.run(main())