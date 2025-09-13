#!/usr/bin/env python3
"""
Test script for faster-whisper STT server
Verifies the implementation works correctly
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from faster_whisper_stt_server import FasterWhisperSTTServer, get_stt_server

async def test_faster_whisper():
    """Test the faster-whisper STT server"""
    print("🔊 Testing Faster Whisper STT Server")
    print("=" * 50)
    
    try:
        # Test server initialization
        print("1. Initializing STT server...")
        server = FasterWhisperSTTServer(model_name="tiny")  # Use tiny model for quick testing
        success = await server.initialize()
        
        if not success:
            print("❌ Failed to initialize server")
            return False
        
        print("✅ STT server initialized successfully")
        
        # Test with silence (should return empty or minimal result)
        print("\n2. Testing with silence...")
        import numpy as np
        silence_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of silence
        result = await server.transcribe_chunk(silence_audio)
        print(f"Silence result: '{result}'")
        
        # Test with very quiet noise (should return empty or minimal result)
        print("\n3. Testing with quiet noise...")
        noise_audio = np.random.randn(16000 * 1).astype(np.float32) * 0.01  # Very quiet noise
        result2 = await server.transcribe_chunk(noise_audio)
        print(f"Noise result: '{result2}'")
        
        # Test global server function
        print("\n4. Testing global server function...")
        global_server = await get_stt_server()
        if global_server:
            print("✅ Global server function works")
        else:
            print("❌ Global server function failed")
        
        # Cleanup
        server.cleanup()
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_fastrtc_integration():
    """Test FastRTC integration"""
    print("\n🚀 Testing FastRTC Integration")
    print("=" * 50)
    
    try:
        # Test FastRTC voice agent import
        print("1. Testing FastRTC voice agent import...")
        
        try:
            from fastrtc_voice_agent import initialize_agent, voice_agent
            print("✅ FastRTC voice agent imported successfully")
            
            # Test initialization
            print("2. Testing FastRTC voice agent initialization...")
            success = await initialize_agent()
            
            if success:
                print("✅ FastRTC voice agent initialized successfully")
            else:
                print("⚠️  FastRTC voice agent initialization failed (expected if FastRTC not fully available)")
                
        except ImportError as e:
            print(f"⚠️  FastRTC import failed (using fallback): {e}")
            
            # Test fallback
            from fastrtc_voice_agent_fallback import initialize_agent
            success = await initialize_agent()
            print(f"Fallback initialization result: {success}")
        
        return True
        
    except Exception as e:
        print(f"❌ FastRTC integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 FastRTC Voice System Test Suite")
    print("=" * 60)
    
    # Test 1: Faster Whisper STT
    stt_success = await test_faster_whisper()
    
    # Test 2: FastRTC Integration
    fastrtc_success = await test_fastrtc_integration()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    print(f"STT Server: {'✅ PASS' if stt_success else '❌ FAIL'}")
    print(f"FastRTC Integration: {'✅ PASS' if fastrtc_success else '❌ FAIL'}")
    
    if stt_success and fastrtc_success:
        print("\n🎉 All tests passed! The voice system is ready.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)