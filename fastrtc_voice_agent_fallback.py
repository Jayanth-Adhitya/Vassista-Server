#!/usr/bin/env python3
"""
FastRTC Voice Agent Fallback - Stub implementation when FastRTC is not available
This allows the server to start normally without FastRTC dependencies
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockVoiceAgent:
    def __init__(self):
        self.stt_server = None
        self.tts_server = None
        
    async def initialize(self):
        logger.warning("FastRTC not available - using mock voice agent")
        return False
    
    async def process_voice_input(self, transcript: str) -> str:
        return "FastRTC voice system not available"

class MockMobileContextManager:
    def __init__(self):
        self.context_data = {}
    
    async def update_context(self, context_data):
        self.context_data = context_data

# Global instances (fallback)
voice_agent = MockVoiceAgent()
mobile_context = MockMobileContextManager()

# Stub functions
async def initialize_agent():
    """Initialize the FastRTC voice agent (fallback)"""
    logger.warning("FastRTC dependencies not available - using fallback mode")
    return False

def get_fastrtc_stream():
    """Get FastRTC stream (fallback)"""
    logger.warning("FastRTC stream not available")
    return None

async def update_mobile_context(context_data):
    """Update mobile context (fallback)"""
    await mobile_context.update_context(context_data)

if __name__ == "__main__":
    async def main():
        logger.info("FastRTC Voice Agent Fallback - testing")
        success = await initialize_agent()
        logger.info(f"Initialization result: {success}")
    
    asyncio.run(main())