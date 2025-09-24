"""
Simple WebSocket Audio Relay - Fast voice processing without WebRTC complexity
Handles audio streaming via WebSocket for reliable real-time voice conversation
"""

import asyncio
import logging
import json
import numpy as np
from typing import Optional, Dict, Any
import base64
import io
import socket
import ipaddress

# Voice processing imports
try:
    from faster_whisper_stt_server import FasterWhisperSTTServer
    from kokoro_tts_server import KokoroTTSServer
    VOICE_SERVICES_AVAILABLE = True
except ImportError:
    VOICE_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleAudioRelay:
    """Simple audio relay that handles WebSocket audio streaming"""

    def __init__(self, stt_server: Optional[FasterWhisperSTTServer] = None,
                 tts_server: Optional[KokoroTTSServer] = None):
        self.stt_server = stt_server
        self.tts_server = tts_server
        self.active_rooms: Dict[str, Dict] = {}
        self.audio_buffers: Dict[str, bytes] = {}
        self.streaming_config: Dict[str, Dict] = {}
        self.context_data: Dict[str, Any] = {}

        logger.info("🎵 Simple Audio Relay initialized")

    async def handle_audio_chunk(self, room: str, audio_data: bytes, websocket, is_final: bool = False):
        """Handle incoming audio chunk via WebSocket"""
        try:
            logger.info(f"🎤 Received audio chunk for room: {room}, size: {len(audio_data)} bytes, is_final: {is_final}")

            # Store audio data in buffer for processing
            if room not in self.audio_buffers:
                self.audio_buffers[room] = b""

            self.audio_buffers[room] += audio_data

            # Only process if this is marked as final (complete recording)
            if is_final:
                logger.info(f"🎯 Final chunk received, processing {len(self.audio_buffers[room])} bytes total")
                # Process the accumulated audio
                await self.process_audio_buffer(room, websocket)
            else:
                logger.info(f"📦 Buffering chunk, total buffer size: {len(self.audio_buffers[room])} bytes")

        except Exception as e:
            logger.error(f"❌ Error handling audio chunk: {e}")

    async def process_audio_buffer(self, room: str, websocket):
        """Process accumulated audio buffer"""
        try:
            if room not in self.audio_buffers or not self.audio_buffers[room]:
                return

            audio_data = self.audio_buffers[room]
            self.audio_buffers[room] = b""  # Clear buffer

            logger.info(f"🎵 Processing audio buffer: {len(audio_data)} bytes")

            # Save received audio to WAV file for debugging
            try:
                import wave
                import time
                timestamp = int(time.time())
                wav_filename = f"received_audio_{room}_{timestamp}.wav"

                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(audio_data)

                logger.info(f"💾 Saved received audio to: {wav_filename}")
            except Exception as save_error:
                logger.error(f"❌ Failed to save audio file: {save_error}")

            # Convert bytes to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Use STT to transcribe
            if self.stt_server:
                transcript = await self.stt_server.transcribe_audio_array(audio_array, sample_rate=16000)
                if transcript and transcript.strip():
                    logger.info(f"📝 Transcribed: {transcript}")

                    # Generate AI response
                    ai_response = await self._generate_ai_response(transcript)
                    if ai_response:
                        logger.info(f"🤖 AI Response: {ai_response}")

                        # Generate streaming TTS audio
                        if self.tts_server:
                            logger.info(f"🎵 Starting streaming TTS for: {ai_response}")

                            # Send TTS start event
                            start_message = {
                                "type": "tts_start",
                                "room": room,
                                "data": {
                                    "text": ai_response,
                                    "total_text_length": len(ai_response)
                                }
                            }
                            await websocket.send(json.dumps(start_message))
                            logger.info(f"📤 TTS streaming started for room {room}")

                            # Stream TTS chunks
                            chunk_count = 0
                            total_chunks = 0

                            try:
                                # Use the streaming TTS functionality
                                async for audio_base64_chunk in self.tts_server.text_to_base64_streaming(ai_response):
                                    if audio_base64_chunk and len(audio_base64_chunk) > 0:
                                        chunk_count += 1

                                        # Send TTS chunk
                                        chunk_message = {
                                            "type": "tts_chunk",
                                            "room": room,
                                            "data": {
                                                "audio": audio_base64_chunk,
                                                "format": "wav",
                                                "sample_rate": 24000,
                                                "chunk_index": chunk_count,
                                                "is_final": False
                                            }
                                        }

                                        await websocket.send(json.dumps(chunk_message))
                                        logger.info(f"📤 TTS chunk {chunk_count} sent to room {room} ({len(audio_base64_chunk)} chars)")

                                        # Small delay to prevent overwhelming the client
                                        await asyncio.sleep(0.01)

                                total_chunks = chunk_count

                                # Send TTS complete event
                                complete_message = {
                                    "type": "tts_complete",
                                    "room": room,
                                    "data": {
                                        "text": ai_response,
                                        "total_chunks": total_chunks,
                                        "success": True
                                    }
                                }
                                await websocket.send(json.dumps(complete_message))
                                logger.info(f"✅ TTS streaming completed for room {room} ({total_chunks} chunks)")

                            except Exception as streaming_error:
                                logger.error(f"❌ TTS streaming error: {streaming_error}")

                                # Send error message
                                error_message = {
                                    "type": "tts_error",
                                    "room": room,
                                    "data": {
                                        "error": str(streaming_error),
                                        "chunks_sent": chunk_count
                                    }
                                }
                                await websocket.send(json.dumps(error_message))
                        else:
                            logger.warning("⚠️ TTS server not available")

        except Exception as e:
            logger.error(f"❌ Error processing audio buffer: {e}")

    async def _generate_ai_response(self, transcript: str) -> Optional[str]:
        """Generate AI response to the transcript"""
        try:
            # Simple response generation
            response = f"I heard you say: '{transcript}'. How can I help you?"
            return response
        except Exception as e:
            logger.error(f"❌ AI response generation error: {e}")
            return None

    def get_valid_local_ips(self):
        """Get valid local IP addresses, filtering out APIPA and invalid addresses"""
        valid_ips = []
        invalid_ip_ranges = [
            ipaddress.ip_network('169.254.0.0/16'),  # APIPA range
            ipaddress.ip_network('0.0.0.0/8'),       # Invalid range
            ipaddress.ip_network('127.0.0.0/8'),     # Loopback
            ipaddress.ip_network('192.0.2.0/24'),    # Test network
            ipaddress.ip_network('198.51.100.0/24'), # Test network
            ipaddress.ip_network('203.0.113.0/24'),  # Test network
        ]

        try:
            # Get local hostname
            hostname = socket.gethostname()
            logger.info(f"🔍 Getting IP addresses for hostname: {hostname}")

            # Get all IP addresses for this host
            addresses = socket.getaddrinfo(hostname, None)

            for addr_info in addresses:
                family, socktype, proto, canonname, sockaddr = addr_info
                ip = sockaddr[0]

                try:
                    ip_obj = ipaddress.ip_address(ip)

                    # Skip invalid IP ranges
                    is_invalid = any(ip_obj in invalid_range for invalid_range in invalid_ip_ranges)

                    if not is_invalid and isinstance(ip_obj, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
                        if ip not in valid_ips:
                            valid_ips.append(ip)
                            logger.info(f"✅ Found valid IP: {ip}")

                except ValueError:
                    logger.warning(f"⚠️ Invalid IP format: {ip}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error getting local IP addresses: {e}")

        # If no valid IPs found, try to get a specific interface
        if not valid_ips:
            try:
                # Try to get the IP for a common interface
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))  # Connect to Google DNS
                    local_ip = s.getsockname()[0]
                    if local_ip not in valid_ips:
                        valid_ips.append(local_ip)
                        logger.info(f"✅ Found fallback IP: {local_ip}")
            except Exception as e:
                logger.error(f"❌ Error getting fallback IP: {e}")

        logger.info(f"📋 Valid local IPs found: {valid_ips}")
        return valid_ips

    async def join_room_as_agent(self, signaling_server, room: str):
        """Join a room as the voice processing agent"""
        try:
            logger.info(f"🤖 Voice agent joining room: {room}")

            # Create agent session for this room
            self.active_rooms[room] = {
                'status': 'active',
                'clients': set(),
                'processing': False
            }

            return True

        except Exception as e:
            logger.error(f"❌ Error joining room as agent: {e}")
            return False

    async def handle_webrtc_offer(self, room: str, offer_data: Dict):
        """Handle WebRTC offer from mobile client"""
        try:
            logger.info(f"📨 Processing WebRTC offer for room: {room}")

            # Create a WebRTC peer connection that will work through the signaling server
            # Since direct P2P connection is failing, we'll use the signaling server as relay
            ice_servers = [
                RTCIceServer(urls=['stun:stun.l.google.com:19302']),
                RTCIceServer(urls=['stun:stun1.l.google.com:19302']),
            ]

            # Configure RTC with network interface restrictions to avoid APIPA addresses
            configuration = RTCConfiguration(iceServers=ice_servers)

            pc = RTCPeerConnection(configuration=configuration)

            # Add custom ICE gathering options to filter out APIPA addresses
            @pc.on("icegatheringstatechange")
            def on_ice_gathering_state_change():
                if pc.iceGatheringState == "complete":
                    logger.info("🧊 ICE gathering completed successfully")
                elif pc.iceGatheringState == "gathering":
                    logger.info("🧊 ICE gathering started...")

            # Add ICE candidate filtering to prevent APIPA addresses
            @pc.on("icecandidate")
            def on_ice_candidate(candidate):
                if candidate:
                    try:
                        # Parse candidate IP address
                        candidate_parts = candidate.candidate.split()
                        if len(candidate_parts) >= 5:
                            candidate_ip = candidate_parts[4]

                            # Check if it's an APIPA address
                            try:
                                ip_obj = ipaddress.ip_address(candidate_ip)
                                is_apipa = ipaddress.ip_network('169.254.0.0/16').supernet_of(ip_obj)

                                if is_apipa:
                                    logger.warning(f"🚫 Filtering out APIPA candidate: {candidate_ip}")
                                    return  # Skip this candidate

                            except ValueError:
                                # Invalid IP format, let it through
                                pass

                        logger.info(f"✅ ICE candidate approved: {candidate_ip}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing ICE candidate: {e}")
            self.peer_connections[room] = pc

            # Set up audio track handling for media relay approach
            @pc.on("track")
            def on_track(track):
                logger.info(f"🎤 Received audio track: {track.kind}")
                if track.kind == "audio":
                    logger.info(f"🎧 Starting audio processing for room: {room}")

                    # Start a background task to process audio frames
                    asyncio.create_task(self._process_audio_track(room, track))

            # Add audio track processing as a separate async method
            async def _process_audio_track(self, room: str, track):
                """Process audio frames from the track"""
                try:
                    while True:
                        # Receive audio frame
                        frame = await track.recv()

                        if frame:
                            # Convert audio frame to numpy array for processing
                            if hasattr(frame, 'to_ndarray'):
                                audio_data = frame.to_ndarray()
                                if len(audio_data.shape) == 2:
                                    # Convert stereo to mono
                                    audio_data = audio_data.mean(axis=1)

                                # Process with VAD and voice pipeline
                                await self.process_audio_chunk(room, audio_data.astype(np.float32))
                except Exception as e:
                    logger.error(f"❌ Audio track processing error: {e}")

            # Bind the method to self for access
            self._process_audio_track = _process_audio_track.__get__(self, FastRTCVoiceAgent)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"🔗 Connection state changed to: {pc.connectionState}")
                if pc.connectionState == "connected":
                    logger.info(f"✅ WebRTC connection established for room: {room}")

                    # Notify client that WebRTC connection is ready
                    from webrtc_signaling import signaling_server
                    import asyncio
                    asyncio.create_task(signaling_server.send_to_room(room, {
                        'type': 'webrtc_connected',
                        'room': room,
                        'message': 'WebRTC connection established successfully'
                    }))
                elif pc.connectionState == "failed":
                    logger.error(f"❌ WebRTC connection failed for room: {room}")
                elif pc.connectionState == "closed":
                    logger.info(f"🔒 WebRTC connection closed for room: {room}")
                    self.peer_connections.pop(room, None)

            # Set remote description (the offer)
            offer_sdp = RTCSessionDescription(sdp=offer_data.get('sdp', ''), type=offer_data.get('type', 'offer'))
            await pc.setRemoteDescription(offer_sdp)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            logger.info(f"📤 Sending WebRTC answer for room: {room}")
            return {
                'type': answer.type,
                'sdp': answer.sdp
            }

        except Exception as e:
            logger.error(f"❌ Error handling WebRTC offer: {e}")
            # Don't clean up peer connection on error - let it be cleaned up later
            # self.peer_connections.pop(room, None)
            return None

    async def handle_ice_candidate(self, room: str, candidate_data: Dict):
        """Handle ICE candidate from mobile client"""
        try:
            if room not in self.peer_connections:
                logger.warning(f"❌ No peer connection found for room: {room}")
                return

            pc = self.peer_connections[room]

            # Check if we have a valid candidate
            candidate_str = candidate_data.get('candidate', '')
            if not candidate_str or candidate_str.strip() == '':
                logger.debug(f"🧊 Skipping empty ICE candidate for room: {room}")
                return

            logger.info(f"🧊 Processing ICE candidate for room {room}: {candidate_str[:50]}...")

            # Parse the candidate string manually and create RTCIceCandidate
            # Example: "candidate:1643776688 1 udp 2122194688 192.168.29.212 40319 typ host generation 0 ufrag P5Cv network-id 4 network-cost 10"
            parts = candidate_str.split()
            if len(parts) < 8 or not parts[0].startswith('candidate:'):
                logger.warning(f"❌ Invalid candidate format: {candidate_str}")
                return

            foundation = parts[0].split(':')[1]
            component = int(parts[1])
            protocol = parts[2].lower()
            priority = int(parts[3])
            ip = parts[4]
            port = int(parts[5])
            cand_type = parts[7]  # host, srflx, relay, etc.

            candidate = RTCIceCandidate(
                component=component,
                foundation=foundation,
                ip=ip,
                port=port,
                priority=priority,
                protocol=protocol,
                type=cand_type,
                sdpMid=candidate_data.get('sdpMid', '0'),
                sdpMLineIndex=candidate_data.get('sdpMLineIndex', 0)
            )

            await pc.addIceCandidate(candidate)
            logger.info(f"✅ Added ICE candidate for room: {room}")

        except Exception as e:
            logger.error(f"❌ Error handling ICE candidate: {e}")
            logger.error(f"   Candidate data: {candidate_data}")
            logger.error(f"   Candidate string: '{candidate_str}'")

    async def process_voice_message(self, room: str, audio_data: bytes):
        """Process voice message using STT → AI → TTS pipeline"""
        try:
            if not VOICE_SERVICES_AVAILABLE or not self.stt_server or not self.tts_server:
                logger.warning("Voice services not available")
                return None

            # Mark as processing
            if room in self.active_rooms:
                self.active_rooms[room]['processing'] = True

            logger.info(f"🎤 Processing voice message for room: {room}")

            # 1. Speech-to-Text
            transcript = await self._transcribe_audio(audio_data)
            if not transcript:
                return None

            logger.info(f"📝 Transcript: {transcript}")

            # 2. Generate AI response
            ai_response = await self._generate_ai_response(transcript)
            if not ai_response:
                return None

            logger.info(f"🤖 AI Response: {ai_response}")

            # 3. Text-to-Speech
            audio_response = await self._synthesize_speech(ai_response)
            if not audio_response:
                return None

            # Mark as done processing
            if room in self.active_rooms:
                self.active_rooms[room]['processing'] = False

            return {
                'transcript': transcript,
                'response_text': ai_response,
                'response_audio': audio_response
            }

        except Exception as e:
            logger.error(f"❌ Error processing voice message: {e}")
            if room in self.active_rooms:
                self.active_rooms[room]['processing'] = False
            return None

    async def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using Faster Whisper"""
        try:
            # Convert audio bytes to numpy array
            # This is a simplified version - in reality you'd need proper audio format handling
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            # Use STT server
            result = await self.stt_server.transcribe_sync(audio_array)
            return result.strip() if result else None

        except Exception as e:
            logger.error(f"❌ STT error: {e}")
            return None

    async def _generate_ai_response(self, transcript: str) -> Optional[str]:
        """Generate AI response to the transcript"""
        try:
            # For now, create a simple response
            # In a full implementation, this would call your AI model
            context_info = ""
            if self.context_data:
                context_info = f" (I have access to your {len(self.context_data.get('notifications', []))} notifications and {len(self.context_data.get('sms_messages', []))} messages)"

            response = f"I heard you say: '{transcript}'. How can I help you?{context_info}"
            return response

        except Exception as e:
            logger.error(f"❌ AI response generation error: {e}")
            return None

    async def _synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Kokoro TTS"""
        try:
            # Use TTS server
            audio_data = await self.tts_server.synthesize(text)
            if len(audio_data) > 0:
                # Convert numpy array to bytes
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                return audio_bytes
            return None

        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            return None

    def update_context(self, context_data: Dict[str, Any]):
        """Update context data for AI responses"""
        self.context_data = context_data
        logger.info(f"📱 Context updated: {len(context_data.get('notifications', []))} notifications, {len(context_data.get('sms_messages', []))} messages")

    def get_room_status(self, room: str) -> Dict[str, Any]:
        """Get status of a room"""
        if room in self.active_rooms:
            return self.active_rooms[room]
        return {'status': 'inactive'}

    async def leave_room(self, room: str):
        """Leave a room and cleanup"""
        if room in self.active_rooms:
            # Stop any active audio streams
            if hasattr(self, 'streaming_rooms') and room in self.streaming_rooms:
                await self.stop_audio_stream(room)

            del self.active_rooms[room]

        # Close peer connection
        if room in self.peer_connections:
            pc = self.peer_connections[room]
            await pc.close()
            del self.peer_connections[room]
            logger.info(f"🔒 Closed WebRTC peer connection for room: {room}")

        logger.info(f"🚪 Voice agent left room: {room}")

    async def start_audio_stream(self, room: str, stream_config: Dict[str, Any]):
        """Start real-time audio streaming for a room with VAD"""
        try:
            logger.info(f"🎤 Starting audio stream for room: {room}")

            # Initialize streaming rooms dict if not exists
            if not hasattr(self, 'streaming_rooms'):
                self.streaming_rooms = {}

            # Configure streaming parameters with VAD - more sensitive settings
            self.streaming_config = {
                "sample_rate": stream_config.get("sample_rate", 16000),
                "channels": stream_config.get("channels", 1),
                "chunk_duration_ms": stream_config.get("chunk_duration_ms", 250),
                "enable_vad": stream_config.get("enable_vad", True),
                "vad_threshold": 0.3,  # Lower threshold for better speech detection
                "buffer_size": stream_config.get("chunk_duration_ms", 250) * 16,  # Buffer size in samples
                "is_active": True,
                "audio_buffer": np.array([], dtype=np.float32),
                "last_activity": 0,
            }

            self.streaming_rooms[room] = self.streaming_config

            logger.info(f"✅ Audio streaming configured for room {room} with VAD enabled")
            logger.info(f"📊 Config: {self.streaming_config['sample_rate']}Hz, {self.streaming_config['chunk_duration_ms']}ms chunks, VAD threshold: {self.streaming_config['vad_threshold']}")

        except Exception as e:
            logger.error(f"❌ Error starting audio stream for room {room}: {e}")
            raise

    async def stop_audio_stream(self, room: str):
        """Stop real-time audio streaming for a room"""
        try:
            if hasattr(self, 'streaming_rooms') and room in self.streaming_rooms:
                self.streaming_rooms[room]["is_active"] = False
                del self.streaming_rooms[room]
                logger.info(f"🔇 Audio streaming stopped for room: {room}")
            else:
                logger.warning(f"⚠️ No active audio stream found for room: {room}")

        except Exception as e:
            logger.error(f"❌ Error stopping audio stream for room {room}: {e}")

    async def process_audio_chunk(self, room: str, audio_chunk: np.ndarray):
        """Process real-time audio chunk with VAD"""
        try:
            if not hasattr(self, 'streaming_rooms') or room not in self.streaming_rooms:
                logger.warning(f"⚠️ No streaming config for room: {room}")
                return

            config = self.streaming_rooms[room]
            if not config["is_active"]:
                return

            # Add chunk to buffer
            config["audio_buffer"] = np.concatenate([config["audio_buffer"], audio_chunk])

            # Process with VAD if we have enough audio (at least 500ms for VAD)
            min_vad_samples = int(0.5 * config["sample_rate"])  # 500ms
            if len(config["audio_buffer"]) >= min_vad_samples:

                # Use STT server's VAD capabilities
                if self.stt_server and config["enable_vad"]:
                    # Transcribe with VAD - this will only return results if speech is detected
                    transcript = await self._transcribe_audio_with_vad(
                        config["audio_buffer"],
                        sample_rate=config["sample_rate"],
                        use_vad=True
                    )

                    if transcript and transcript.strip():
                        logger.info(f"🎯 VAD detected speech: '{transcript}'")

                        # Generate AI response
                        ai_response = await self._generate_ai_response(transcript)

                        if ai_response:
                            # Generate TTS audio
                            tts_audio = await self._synthesize_speech(ai_response)

                            if tts_audio:
                                logger.info(f"🔊 Generated TTS response: {len(tts_audio)} bytes")
                                # Here we would send the TTS audio back via WebRTC
                                # For now, we'll emit it through the signaling system
                                await self._send_tts_response(room, ai_response, tts_audio)

                        # Clear buffer after processing
                        config["audio_buffer"] = np.array([], dtype=np.float32)
                    else:
                        # VAD detected no speech - keep last 250ms for continuity
                        keep_samples = int(0.25 * config["sample_rate"])
                        if len(config["audio_buffer"]) > keep_samples:
                            config["audio_buffer"] = config["audio_buffer"][-keep_samples:]

        except Exception as e:
            logger.error(f"❌ Error processing audio chunk for room {room}: {e}")

    async def _transcribe_audio_with_vad(self, audio_data: np.ndarray, sample_rate: int = 16000, use_vad: bool = True) -> Optional[str]:
        """Transcribe audio using faster-whisper with VAD"""
        try:
            if self.stt_server:
                # Use the STT server's transcribe method with VAD enabled
                transcript = await self.stt_server.transcribe_audio_array(
                    audio_data,
                    sample_rate=sample_rate,
                    use_vad=use_vad
                )
                return transcript
            else:
                logger.warning("STT server not available")
                return None

        except Exception as e:
            logger.error(f"❌ VAD transcription error: {e}")
            return None

    async def _send_tts_response(self, room: str, text_response: str, tts_audio: bytes):
        """Send TTS response back to the client via signaling"""
        try:
            # For now, we'll send via the signaling system
            # In a full WebRTC implementation, this would go through the RTC data channel
            from webrtc_signaling import signaling_server

            # Convert audio to base64 for transmission
            audio_base64 = base64.b64encode(tts_audio).decode('utf-8')

            response_message = {
                "type": "audio_response",
                "room": room,
                "data": {
                    "text": text_response,
                    "audio": audio_base64,
                    "format": "pcm_16bit",
                    "sample_rate": 24000  # Kokoro TTS output rate
                },
                "sender_id": "voice_agent",
                "timestamp": asyncio.get_event_loop().time()
            }

            logger.info(f"📤 Broadcasting TTS response to room {room}")
            logger.info(f"📝 Response text: {text_response}")
            logger.info(f"🔊 Audio size: {len(audio_base64)} characters")

            # Broadcast to room
            await signaling_server.broadcast_to_room(room, response_message)
            logger.info(f"✅ TTS response broadcasted to room {room}")

        except Exception as e:
            logger.error(f"❌ Error sending TTS response: {e}")
            logger.error(f"❌ Error details: {str(e)}")

# Global voice agent instance
audio_relay: Optional[SimpleAudioRelay] = None

def get_audio_relay() -> Optional[SimpleAudioRelay]:
    """Get the global audio relay instance"""
    return audio_relay

def initialize_audio_relay(stt_server=None, tts_server=None):
    """Initialize the global audio relay"""
    global audio_relay
    audio_relay = SimpleAudioRelay(stt_server, tts_server)
    return audio_relay
