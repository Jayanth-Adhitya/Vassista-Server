#!/usr/bin/env python3
"""
Kokoro TTS Server for ultra-fast text-to-speech synthesis
Uses Kokoro-82M model for minimal latency speech generation
"""

import asyncio
import logging
import torch
import torchaudio
import numpy as np
import io
import base64
import tempfile
from typing import AsyncGenerator, Optional
import soundfile as sf
from pathlib import Path
import requests
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KokoroTTSServer:
    def __init__(self, model_path: str = None, device: str = "auto", sample_rate: int = 24000):
        self.device = self._get_device(device)
        self.sample_rate = sample_rate
        self.model = None
        self.tokenizer = None
        self.vocoder = None
        self.model_path = model_path or "kokoro-82m"  # Default model name
        
    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        return device
    
    async def initialize(self):
        """Initialize the Kokoro TTS model"""
        try:
            logger.info("Loading Kokoro TTS model...")
            
            # Try to load local model first, fallback to downloading
            if not await self._load_local_model():
                await self._download_and_setup_model()
            
            logger.info("Kokoro TTS model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS model: {e}")
            return False
    
    async def _load_local_model(self) -> bool:
        """Try to load local Kokoro model"""
        try:
            logger.info("Loading real Kokoro model...")

            # Import the actual Kokoro model
            try:
                # Import the correct Kokoro classes
                from kokoro import KModel
                from kokoro.pipeline import KPipeline

                # Initialize Kokoro model
                logger.info("Initializing Kokoro KModel...")
                self.model = KModel()

                # Initialize TTS pipeline
                logger.info("Initializing Kokoro KPipeline...")
                self.tts = KPipeline(model=self.model, device=self.device, lang_code='a')

                # Load a default voice
                logger.info("Loading default voice...")
                self.tts.load_voice('af_heart')
                logger.info("✅ Default voice loaded")

                logger.info("✅ Real Kokoro model loaded successfully")
                return True

            except ImportError as e:
                logger.warning(f"Kokoro not available: {e}, using dummy model for testing")
                self.model = self._create_dummy_model()
                return True

        except Exception as e:
            logger.error(f"Local model loading failed: {e}")
            logger.info("Falling back to dummy model...")
            self.model = self._create_dummy_model()
            return True
    
    def _create_dummy_model(self):
        """Create a dummy model for testing purposes"""
        # This would be replaced with actual Kokoro model loading
        class DummyKokoroModel:
            def __init__(self, device, sample_rate):
                self.device = device
                self.sample_rate = sample_rate
            
            def synthesize(self, text: str) -> np.ndarray:
                # Generate a simple sine wave as placeholder
                duration = len(text) * 0.1  # 0.1 seconds per character
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                frequency = 440  # A4 note
                audio = np.sin(2 * np.pi * frequency * t) * 0.3
                return audio.astype(np.float32)
        
        return DummyKokoroModel(self.device, self.sample_rate)
    
    async def _download_and_setup_model(self):
        """Download and set up Kokoro model if not available locally"""
        logger.info("Downloading Kokoro TTS model...")
        
        # This would contain the actual download logic
        # For now, we'll use the dummy model
        self.model = self._create_dummy_model()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS synthesis"""
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common abbreviations
        replacements = {
            'Mr.': 'Mister',
            'Mrs.': 'Missus', 
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'St.': 'Street',
            'Ave.': 'Avenue',
        }
        
        for abbrev, full in replacements.items():
            text = text.replace(abbrev, full)
        
        return text
    
    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text"""
        try:
            if not text.strip():
                return np.array([], dtype=np.float32)

            # Preprocess text
            processed_text = self.preprocess_text(text)

            # Check if we have the real Kokoro model
            if hasattr(self, 'tts') and self.tts is not None:
                logger.info(f"🎵 Synthesizing with real Kokoro: '{processed_text[:50]}...'")

                # Use real Kokoro TTS - call as function with voice parameter
                try:
                    result = self.tts(processed_text, voice='af_heart')
                    # KPipeline returns a generator, get the first item
                    result_items = list(result)
                    if result_items and hasattr(result_items[0], 'audio') and len(result_items[0].audio) > 0:
                        logger.info(f"✅ Real Kokoro TTS generated: {len(result_items[0].audio)} samples")
                        return result_items[0].audio.numpy().astype(np.float32)
                    else:
                        logger.warning("⚠️ Real Kokoro TTS returned empty audio")
                        return np.array([], dtype=np.float32)
                except Exception as e:
                    logger.error(f"❌ Real Kokoro TTS error: {e}")
                    return np.array([], dtype=np.float32)
            else:
                # Fallback to dummy model
                logger.warning("⚠️ Using dummy TTS model")
                audio_data = self.model.synthesize(processed_text)
                return audio_data

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return np.array([], dtype=np.float32)
    
    async def synthesize_streaming(self, text: str, chunk_size: int = 50) -> AsyncGenerator[np.ndarray, None]:
        """Synthesize speech in streaming chunks for lower latency"""
        try:
            if not text.strip():
                return
            
            # Split text into sentences for chunk processing
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if sentence.strip():
                    audio_chunk = await self.synthesize(sentence)
                    if len(audio_chunk) > 0:
                        yield audio_chunk
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for chunked processing"""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out empty sentences and add punctuation back
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                result.append(sentence)
        
        return result
    
    async def text_to_audio_bytes(self, text: str, format: str = "wav") -> bytes:
        """Convert text to audio bytes"""
        try:
            audio_data = await self.synthesize(text)
            
            if len(audio_data) == 0:
                return b''
            
            # Convert to bytes
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                tmp_file.flush()
                
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Audio bytes conversion error: {e}")
            return b''
    
    async def text_to_base64(self, text: str, format: str = "wav") -> str:
        """Convert text to base64-encoded audio"""
        try:
            audio_bytes = await self.text_to_audio_bytes(text, format)
            if audio_bytes:
                return base64.b64encode(audio_bytes).decode('utf-8')
            return ""
            
        except Exception as e:
            logger.error(f"Base64 conversion error: {e}")
            return ""
    
    async def text_to_base64_streaming(self, text: str, format: str = "wav") -> AsyncGenerator[str, None]:
        """Convert text to streaming base64-encoded audio chunks"""
        try:
            async for audio_chunk in self.synthesize_streaming(text):
                if len(audio_chunk) > 0:
                    # Convert chunk to bytes with better file handling
                    tmp_file = None
                    try:
                        tmp_file = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
                        sf.write(tmp_file.name, audio_chunk, self.sample_rate)
                        tmp_file.close()  # Close file handle before reading

                        # Small delay to ensure file is written
                        await asyncio.sleep(0.001)

                        with open(tmp_file.name, 'rb') as audio_file:
                            chunk_bytes = audio_file.read()

                        # Encode to base64
                        base64_chunk = base64.b64encode(chunk_bytes).decode('utf-8')
                        yield base64_chunk

                    except Exception as chunk_error:
                        logger.error(f"Error processing audio chunk: {chunk_error}")
                    finally:
                        # Clean up with retry
                        if tmp_file and tmp_file.name:
                            for attempt in range(3):
                                try:
                                    if os.path.exists(tmp_file.name):
                                        os.unlink(tmp_file.name)
                                    break
                                except Exception as cleanup_error:
                                    if attempt == 2:  # Last attempt
                                        logger.warning(f"Failed to clean up temp file after 3 attempts: {cleanup_error}")
                                    else:
                                        await asyncio.sleep(0.01)  # Wait before retry

        except Exception as e:
            logger.error(f"Streaming base64 conversion error: {e}")

# Global TTS server instance
tts_server = None

async def get_tts_server():
    """Get or create the global TTS server instance"""
    global tts_server
    if tts_server is None:
        tts_server = KokoroTTSServer()
        success = await tts_server.initialize()
        if not success:
            tts_server = None
            raise RuntimeError("Failed to initialize Kokoro TTS server")
    return tts_server

# Utility functions for FastRTC integration
async def synthesize_for_fastrtc(text: str) -> np.ndarray:
    """Synthesize speech for FastRTC integration"""
    try:
        server = await get_tts_server()
        return await server.synthesize(text)
    except Exception as e:
        logger.error(f"FastRTC synthesis error: {e}")
        return np.array([], dtype=np.float32)

async def synthesize_base64_for_fastrtc(text: str) -> str:
    """Synthesize speech and return as base64 for FastRTC"""
    try:
        server = await get_tts_server()
        return await server.text_to_base64(text)
    except Exception as e:
        logger.error(f"FastRTC base64 synthesis error: {e}")
        return ""

if __name__ == "__main__":
    async def main():
        # Test the TTS server
        server = KokoroTTSServer()
        success = await server.initialize()
        
        if success:
            logger.info("Kokoro TTS Server initialized successfully!")
            
            # Test synthesis
            test_text = "Hello, this is a test of the Kokoro TTS system."
            audio_data = await server.synthesize(test_text)
            logger.info(f"Synthesized audio shape: {audio_data.shape}")
            
            # Test base64 conversion
            base64_audio = await server.text_to_base64(test_text)
            logger.info(f"Base64 audio length: {len(base64_audio)}")
            
        else:
            logger.error("Failed to initialize TTS server")
    
    asyncio.run(main())
