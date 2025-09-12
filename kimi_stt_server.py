#!/usr/bin/env python3
"""
Kimi-Audio STT Server for real-time speech-to-text processing
Uses Moonshot AI's Kimi-Audio model for high-quality speech recognition
"""

import asyncio
import logging
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
from typing import AsyncGenerator
import tempfile
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KimiSTTServer:
    def __init__(self, model_name="moonshotai/Kimi-Audio-7B-Instruct", device="auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.sample_rate = 16000  # Standard sample rate for speech recognition
        
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
        """Initialize the Kimi-Audio model and processor"""
        try:
            logger.info(f"Loading Kimi-Audio model: {self.model_name}")
            
            # Load processor and model with trust_remote_code=True
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Kimi-Audio model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kimi-Audio model: {e}")
            return False
    
    def preprocess_audio(self, audio_data: np.ndarray, original_sr: int = None) -> torch.Tensor:
        """Preprocess audio data for the model"""
        try:
            # Convert to torch tensor if numpy array
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                audio_tensor = audio_data.float()
            
            # Ensure single channel
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            
            # Resample if necessary
            if original_sr and original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sr, 
                    new_freq=self.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            
            return audio_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return None
    
    async def transcribe_chunk(self, audio_chunk: np.ndarray, original_sr: int = None) -> str:
        """Transcribe a single audio chunk"""
        try:
            # Preprocess audio
            audio_tensor = self.preprocess_audio(audio_chunk, original_sr)
            if audio_tensor is None:
                return ""
            
            # Process with the model
            with torch.no_grad():
                # Use the processor to prepare inputs
                inputs = self.processor(
                    audio_tensor.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate transcription
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                
                # Decode the result
                transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                return transcription.strip()
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """Process streaming audio and yield transcriptions"""
        buffer = []
        chunk_duration = 2.0  # Process every 2 seconds
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

# Global STT server instance
stt_server = None

async def get_stt_server():
    """Get or create the global STT server instance"""
    global stt_server
    if stt_server is None:
        stt_server = KimiSTTServer()
        success = await stt_server.initialize()
        if not success:
            stt_server = None
            raise RuntimeError("Failed to initialize Kimi STT server")
    return stt_server

# Utility function for FastRTC integration
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
        server = KimiSTTServer()
        success = await server.initialize()
        
        if success:
            logger.info("Kimi STT Server initialized successfully!")
            
            # Test with a dummy audio chunk
            dummy_audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds of noise
            result = await server.transcribe_chunk(dummy_audio)
            logger.info(f"Test transcription result: {result}")
        else:
            logger.error("Failed to initialize STT server")
    
    asyncio.run(main())