# FastRTC Voice System Setup Guide

## Overview
This guide covers setting up the FastRTC-based voice system that replaces the previous WebSocket + React Native Voice implementation with a low-latency WebRTC solution using local STT/TTS models.

## Architecture
```
React Native (WebRTC) → FastRTC Server → Kimi-Audio STT → AgentZero → Kokoro TTS → Audio Stream
```

**Expected Latency**: ~500ms total (vs previous 3000ms+)

## Backend Setup

### Option 1: Docker Setup (Recommended)

#### With GPU Support (Recommended for best performance)
```bash
cd Server
# Build and run with GPU acceleration
docker-compose -f docker-compose.fastrtc.yml up --build
```

#### CPU-Only Version
```bash
cd Server
# Edit docker-compose.fastrtc.yml to use CPU version (see comments in file)
docker-compose -f docker-compose.fastrtc.yml up --build
```

#### Docker Requirements
- **GPU Version**: NVIDIA Docker runtime, CUDA 11.8+
- **CPU Version**: Docker with 8GB+ RAM allocated
- **Storage**: 20GB+ available space for models

### Option 2: Manual Setup

#### 1. Install Dependencies
```bash
cd Server
pip install -r fastrtc_requirements.txt
```

#### 2. Model Setup

##### Kimi-Audio STT (Moonshot AI)
```bash
# The model will auto-download on first run
# Ensure you have sufficient disk space (~7GB for the model)
python kimi_stt_server.py  # Test STT server
```

##### Kokoro TTS
```bash
# Kokoro model will auto-setup on first run
# Requires ~2GB RAM, GPU recommended for best performance
python kokoro_tts_server.py  # Test TTS server
```

#### 3. Start FastRTC Server
```bash
python main.py
```

The server will:
- Initialize FastRTC voice agent
- Set up Kimi-Audio STT server 
- Set up Kokoro TTS server
- Start FastAPI with FastRTC endpoints

## React Native App Setup

### 1. Install Dependencies
```bash
cd NotificationReaderApp
npm install
# react-native-webrtc will be installed automatically
```

### 2. Android Permissions
The app already includes the necessary permissions in AndroidManifest.xml:
- RECORD_AUDIO
- MODIFY_AUDIO_SETTINGS
- VIBRATE

### 3. Run the App
```bash
npm run android
```

## API Endpoints

### FastRTC Status
```
GET /fastrtc/status
```
Returns the status of FastRTC system components.

### Update Context
```
POST /fastrtc/context
{
  "sms_messages": [...],
  "notifications": [...], 
  "chat_history": [...]
}
```

### Voice Query (Testing)
```
POST /fastrtc/voice-query
{
  "transcript": "Hello, how are you?",
  "context_data": {...}
}
```

### Stream Configuration
```
GET /fastrtc/stream
```
Returns WebRTC configuration for React Native client.

## Performance Optimization

### Server-Side Optimizations

1. **GPU Acceleration**
   - Ensure CUDA is available for Kokoro TTS
   - Use `nvidia-smi` to monitor GPU utilization

2. **Model Loading**
   - Models are loaded once on startup to avoid initialization latency
   - Consider model quantization for memory optimization

3. **Audio Processing**
   - 16kHz sample rate for optimal STT performance
   - Echo cancellation and noise suppression enabled

### Client-Side Optimizations

1. **WebRTC Configuration**
   - Uses Google STUN servers for NAT traversal
   - Optimized audio constraints for low latency

2. **Voice Activity Detection**
   - Built-in silence detection (2 second timeout)
   - Automatic turn-taking to minimize latency

## Testing & Monitoring

### Latency Testing
1. **End-to-End Test**:
   - Speak into the app
   - Measure time from speech start to audio response start
   - Target: <800ms

2. **Component Testing**:
   - STT latency: Test with `kimi_stt_server.py`
   - TTS latency: Test with `kokoro_tts_server.py` 
   - AgentZero latency: Monitor API response times

### Performance Monitoring
- Check `/fastrtc/status` endpoint
- Monitor server logs for processing times
- Use Chrome DevTools for WebRTC stats in browser testing

## Troubleshooting

### Docker Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   
   # If this fails, install NVIDIA Docker runtime:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **Out of Memory**
   ```bash
   # Increase Docker memory allocation to 8GB+ in Docker Desktop
   # Or use CPU-only version for lower memory usage
   ```

3. **Model Download Failures**
   ```bash
   # Check container logs
   docker-compose -f docker-compose.fastrtc.yml logs fastrtc-voice-server
   
   # Ensure internet connectivity from container
   docker exec -it fastrtc-voice-server curl -I https://huggingface.co
   ```

4. **Port Conflicts**
   ```bash
   # Change port mapping in docker-compose.fastrtc.yml
   ports:
     - "8001:8000"  # Use different external port
   ```

### Common Issues

1. **FastRTC Import Error**
   ```bash
   pip install "fastrtc[vad,tts]"
   ```

2. **Model Loading Failures**
   - Ensure sufficient disk space and memory
   - Check internet connection for model downloads
   - Verify Python/PyTorch versions

3. **WebRTC Connection Issues**
   - Check network connectivity
   - Verify STUN server accessibility
   - Review Android audio permissions

4. **High Latency**
   - Monitor individual component latencies
   - Check for GPU acceleration usage
   - Optimize network conditions

### Debug Modes

1. **Enable Verbose Logging**:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test Individual Components**:
   ```bash
   python fastrtc_voice_agent.py  # Test full pipeline
   python kimi_stt_server.py      # Test STT only
   python kokoro_tts_server.py    # Test TTS only
   ```

## Hardware Requirements

### Minimum Requirements
- CPU: 4+ cores, 2.5GHz+
- RAM: 8GB (4GB for models, 4GB for system)
- Storage: 10GB free space
- Network: Stable internet connection

### Recommended Requirements  
- GPU: NVIDIA GTX 1060+ (6GB VRAM)
- CPU: 8+ cores, 3.0GHz+
- RAM: 16GB
- Storage: SSD with 20GB+ free space
- Network: Low-latency connection (<50ms to server)

## Performance Benchmarks

### Expected Latencies
- **STT Processing**: 100-200ms (Kimi-Audio streaming)
- **LLM Processing**: 200-300ms (AgentZero)
- **TTS Synthesis**: 50-100ms first token (Kokoro)
- **Network Overhead**: 20-50ms (local WebRTC)
- **Total End-to-End**: 470-650ms

### Hardware Performance
- **RTX 4090**: ~40-70ms TTS synthesis
- **RTX 3090 Ti**: ~90-100ms TTS synthesis  
- **Modern CPU**: ~300-500ms TTS synthesis
- **Older CPU**: Up to 1000ms TTS synthesis

## Production Considerations

1. **Scalability**
   - Consider model server clustering
   - Implement load balancing for multiple users
   - Monitor resource usage patterns

2. **Reliability**
   - Add fallback mechanisms for model failures
   - Implement automatic restarts for crashed processes
   - Set up health checks and monitoring

3. **Security**
   - Implement proper authentication for FastRTC endpoints
   - Use secure WebRTC configurations
   - Validate and sanitize all inputs

This setup provides a complete low-latency voice interaction system with local models, eliminating the need for external APIs while achieving significantly better performance than the previous implementation.