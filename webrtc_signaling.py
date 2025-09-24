"""
WebRTC Signaling Server for Voice Chat
Using WebSocket for signaling only, WebRTC for actual audio streaming
Based on the working pattern from your video streaming project
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import json
import logging
import asyncio
from dataclasses import dataclass, asdict
import time

# Import voice agent
from fastrtc_voice_agent import get_audio_relay

logger = logging.getLogger(__name__)

@dataclass
class SignalingMessage:
    """WebRTC signaling message"""
    type: str  # offer, answer, ice, join, leave, data, start_voice_stream, stop_voice_stream, audio_chunk
    room: Optional[str] = None
    sdp: Optional[dict] = None
    candidate: Optional[dict] = None
    data: Optional[dict] = None
    audio_config: Optional[dict] = None
    audio_data: Optional[str] = None  # Base64 encoded audio data
    has_audio: Optional[bool] = None  # Whether audio activity is detected
    sender_id: Optional[str] = None
    timestamp: Optional[float] = None
    duration: Optional[float] = None  # Recording duration in milliseconds
    is_final: Optional[bool] = None  # Whether this is a final audio chunk
    chunk_index: Optional[int] = None  # Index of this chunk in sequence
    total_chunks: Optional[int] = None  # Total number of chunks in sequence
    file_name: Optional[str] = None  # Audio file name for upload confirmations

class WebRTCSignalingServer:
    """WebSocket signaling server for WebRTC connections"""

    def __init__(self):
        # Room-based connection management
        self.rooms: Dict[str, Set[WebSocket]] = {}
        self.websocket_to_room: Dict[WebSocket, str] = {}
        self.websocket_to_id: Dict[WebSocket, str] = {}
        self.next_id = 0
        logger.info("WebRTC Signaling Server initialized")

    def generate_id(self) -> str:
        """Generate unique client ID"""
        self.next_id += 1
        return f"client_{self.next_id}"

    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection for signaling"""
        await websocket.accept()
        client_id = self.generate_id()
        self.websocket_to_id[websocket] = client_id

        logger.info(f"✅ WebRTC signaling connection established: {client_id}")

        # Send welcome message with client ID
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "client_id": client_id,
            "timestamp": time.time()
        }))

        try:
            while True:
                # Receive signaling message
                message = await websocket.receive()

                if message["type"] == "websocket.receive":
                    if "text" in message:
                        await self.handle_signaling_message(websocket, message["text"])

        except WebSocketDisconnect:
            await self.handle_disconnect(websocket)
        except Exception as e:
            logger.error(f"Signaling error: {e}")
            await self.handle_disconnect(websocket)

    async def handle_signaling_message(self, websocket: WebSocket, message_text: str):
        """Process WebRTC signaling messages"""
        try:
            data = json.loads(message_text)
            msg = SignalingMessage(**data)
            msg.sender_id = self.websocket_to_id.get(websocket)
            msg.timestamp = time.time()

            logger.info(f"📨 Signaling message: {msg.type} from {msg.sender_id}")

            if msg.type == "join":
                await self.handle_join_room(websocket, msg)

            elif msg.type == "offer":
                await self.handle_webrtc_offer(websocket, msg)

            elif msg.type == "answer":
                await self.relay_to_room(websocket, msg)

            elif msg.type == "ice":
                await self.handle_ice_candidate(websocket, msg)

            elif msg.type == "leave":
                await self.handle_leave_room(websocket)

            elif msg.type == "data":
                # Handle data channel messages
                await self.handle_data_message(websocket, msg)

            elif msg.type == "start_voice_stream":
                await self.handle_start_voice_stream(websocket, msg)

            elif msg.type == "stop_voice_stream":
                await self.handle_stop_voice_stream(websocket, msg)

            elif msg.type == "audio_chunk":
                await self.handle_audio_chunk(websocket, msg)

            elif msg.type == "audio_activity":
                await self.handle_audio_activity(websocket, msg)

            else:
                logger.warning(f"Unknown signaling message type: {msg.type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON in signaling message")
        except Exception as e:
            logger.error(f"Error handling signaling message: {e}")

    async def handle_join_room(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle client joining a room"""
        room = msg.room or "default"

        # Leave previous room if any
        if websocket in self.websocket_to_room:
            await self.handle_leave_room(websocket)

        # Join new room
        if room not in self.rooms:
            self.rooms[room] = set()

        self.rooms[room].add(websocket)
        self.websocket_to_room[websocket] = room

        client_id = self.websocket_to_id[websocket]
        logger.info(f"👥 {client_id} joined room: {room}")

        # Notify client of successful join
        await websocket.send_text(json.dumps({
            "type": "joined",
            "room": room,
            "client_id": client_id,
            "room_size": len(self.rooms[room]),
            "timestamp": time.time()
        }))

        # Notify other clients in room
        await self.broadcast_to_room(room, {
            "type": "peer_joined",
            "client_id": client_id,
            "timestamp": time.time()
        }, exclude=websocket)

    async def handle_leave_room(self, websocket: WebSocket):
        """Handle client leaving a room"""
        if websocket not in self.websocket_to_room:
            return

        room = self.websocket_to_room[websocket]
        client_id = self.websocket_to_id.get(websocket)

        self.rooms[room].discard(websocket)
        del self.websocket_to_room[websocket]

        # Clean up empty rooms
        if not self.rooms[room]:
            del self.rooms[room]
        else:
            # Notify other clients
            await self.broadcast_to_room(room, {
                "type": "peer_left",
                "client_id": client_id,
                "timestamp": time.time()
            })

        logger.info(f"👋 {client_id} left room: {room}")

    async def handle_webrtc_offer(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle WebRTC offer from client and respond with voice agent answer"""
        try:
            room = msg.room or self.websocket_to_room.get(websocket, "default")
            client_id = self.websocket_to_id.get(websocket)

            logger.info(f"📨 Handling WebRTC offer from {client_id} in room {room}")

            # Get audio relay
            audio_relay = get_audio_relay()
            if not audio_relay:
                logger.error("❌ Audio relay not available")
                return

            # Join room as audio relay if not already
            await audio_relay.join_room_as_agent(self, room)

            # Process the offer and get an answer
            offer_data = {
                'type': msg.sdp.get('type') if msg.sdp else 'offer',
                'sdp': msg.sdp.get('sdp') if msg.sdp else ''
            }

            answer = await audio_relay.handle_webrtc_offer(room, offer_data)

            if answer:
                # Send answer back to the client
                await websocket.send_text(json.dumps({
                    "type": "answer",
                    "sender_id": "voice_agent",
                    "sdp": answer,
                    "room": room,
                    "timestamp": time.time()
                }))

                logger.info(f"📤 Sent WebRTC answer to {client_id}")
            else:
                logger.error(f"❌ Failed to create answer for {client_id}")

        except Exception as e:
            logger.error(f"❌ Error handling WebRTC offer: {e}")

    async def handle_ice_candidate(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle ICE candidate from client and pass to voice agent"""
        try:
            room = msg.room or self.websocket_to_room.get(websocket, "default")
            client_id = self.websocket_to_id.get(websocket)

            logger.debug(f"🧊 Handling ICE candidate from {client_id} in room {room}")

            # Get audio relay and pass the ICE candidate
            audio_relay = get_audio_relay()
            if audio_relay and msg.candidate:
                await audio_relay.handle_ice_candidate(room, msg.candidate)

        except Exception as e:
            logger.error(f"❌ Error handling ICE candidate: {e}")

    async def relay_to_room(self, sender: WebSocket, msg: SignalingMessage):
        """Relay signaling message to all other clients in the room"""
        if sender not in self.websocket_to_room:
            logger.warning("Sender not in any room")
            return

        room = self.websocket_to_room[sender]
        message_dict = asdict(msg)
        message_json = json.dumps(message_dict)

        # Send to all other clients in the room
        await self.broadcast_to_room(room, message_dict, exclude=sender)

    async def broadcast_to_room(self, room: str, message: dict, exclude: Optional[WebSocket] = None):
        """Broadcast message to all clients in a room"""
        if room not in self.rooms:
            return

        message_json = json.dumps(message)
        disconnected = []

        for client in self.rooms[room]:
            if client == exclude:
                continue

            try:
                await client.send_text(message_json)
            except:
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            await self.handle_disconnect(client)

    async def handle_disconnect(self, websocket: WebSocket):
        """Handle client disconnect"""
        client_id = self.websocket_to_id.get(websocket, "unknown")
        logger.info(f"🔌 Signaling disconnected: {client_id}")

        # Leave room if in one
        if websocket in self.websocket_to_room:
            await self.handle_leave_room(websocket)

        # Clean up client data
        self.websocket_to_id.pop(websocket, None)

        try:
            await websocket.close()
        except:
            pass

    async def handle_data_message(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle data channel messages for voice processing"""
        try:
            # Forward data message to voice processing endpoint
            from main import call_agent_zero, stt_server, tts_server
            import base64
            import numpy as np

            data_payload = msg.data if hasattr(msg, 'data') else {}
            message_type = data_payload.get("type")

            logger.info(f"Processing data channel message: {message_type}")

            if message_type == "audio":
                # Process audio data
                audio_base64 = data_payload.get("audio")
                if audio_base64:
                    # Decode and process audio
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # Transcribe using STT
                    if stt_server:
                        transcript = await stt_server.transcribe_audio_array(audio_array, sample_rate=16000)
                    else:
                        transcript = "Voice services not available"

                    if transcript and transcript.strip():
                        # Get AI response
                        ai_response = await call_agent_zero(transcript)

                        # Generate TTS
                        if tts_server:
                            audio_output = await tts_server.synthesize(ai_response)
                            audio_response_base64 = base64.b64encode(audio_output.tobytes()).decode('utf-8')
                        else:
                            audio_response_base64 = ""

                        # Send response back through signaling
                        response_msg = {
                            "type": "data_response",
                            "data": {
                                "type": "response",
                                "transcript": transcript,
                                "text": ai_response,
                                "audio": audio_response_base64
                            },
                            "sender_id": "server",
                            "timestamp": time.time()
                        }

                        # Send back to sender
                        await websocket.send_text(json.dumps(response_msg))

            elif message_type == "text":
                # Process text query
                text = data_payload.get("text", "")
                ai_response = await call_agent_zero(text)

                # Generate TTS if requested
                if data_payload.get("tts", False) and tts_server:
                    audio_output = await tts_server.synthesize(ai_response)
                    audio_base64 = base64.b64encode(audio_output.tobytes()).decode('utf-8')
                else:
                    audio_base64 = ""

                response_msg = {
                    "type": "data_response",
                    "data": {
                        "type": "response",
                        "text": ai_response,
                        "audio": audio_base64
                    },
                    "sender_id": "server",
                    "timestamp": time.time()
                }

                # Send back to sender
                await websocket.send_text(json.dumps(response_msg))

        except Exception as e:
            logger.error(f"Error handling data message: {e}")
            # Send error response
            error_msg = {
                "type": "data_response",
                "data": {
                    "type": "error",
                    "error": str(e)
                },
                "sender_id": "server",
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(error_msg))

    async def handle_start_voice_stream(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle request to start real-time voice streaming"""
        try:
            client_id = self.websocket_to_id.get(websocket, "unknown")
            room = msg.room or "default"
            audio_config = msg.audio_config if msg.audio_config else {}

            logger.info(f"🎤 Starting voice stream for {client_id} in room {room}")

            # Get audio relay and configure for streaming
            audio_relay = get_audio_relay()
            if audio_relay:
                # Configure streaming parameters with more sensitive VAD
                stream_config = {
                    "sample_rate": audio_config.get("sample_rate", 16000),
                    "channels": audio_config.get("channels", 1),
                    "format": audio_config.get("format", "pcm"),
                    "enable_vad": True,  # Enable VAD for real-time processing
                    "vad_threshold": 0.3,  # Lower threshold for better speech detection
                    "chunk_duration_ms": 250,  # 250ms chunks for low latency
                }

                # Notify audio relay to start listening for this room
                await audio_relay.start_audio_stream(room, stream_config)

                # Acknowledge to client
                response = {
                    "type": "voice_stream_started",
                    "room": room,
                    "config": stream_config,
                    "sender_id": "server",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(response))

                logger.info(f"✅ Voice streaming started for {client_id}")
            else:
                raise Exception("Audio relay not available")

        except Exception as e:
            logger.error(f"❌ Error starting voice stream: {e}")
            error_response = {
                "type": "voice_stream_error",
                "error": str(e),
                "sender_id": "server",
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(error_response))

    async def handle_stop_voice_stream(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle request to stop real-time voice streaming"""
        try:
            client_id = self.websocket_to_id.get(websocket, "unknown")
            room = msg.room or "default"

            logger.info(f"🔇 Stopping voice stream for {client_id} in room {room}")

            # Get audio relay and stop streaming
            audio_relay = get_audio_relay()
            if audio_relay:
                await audio_relay.stop_audio_stream(room)

                # Acknowledge to client
                response = {
                    "type": "voice_stream_stopped",
                    "room": room,
                    "sender_id": "server",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(response))

                logger.info(f"✅ Voice streaming stopped for {client_id}")
            else:
                logger.warning("Voice agent not available")

        except Exception as e:
            logger.error(f"❌ Error stopping voice stream: {e}")

    async def handle_audio_chunk(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle audio chunk from client for real-time processing"""
        try:
            client_id = self.websocket_to_id.get(websocket, "unknown")
            room = msg.room or "default"

            logger.info(f"🎵 Received audio chunk from {client_id} in room {room}")

            # Get audio relay and process the audio chunk
            audio_relay = get_audio_relay()
            if audio_relay and msg.audio_data:
                # Decode base64 audio data
                import base64
                import numpy as np

                try:
                    audio_bytes = base64.b64decode(msg.audio_data)

                    # Ensure even number of bytes for 16-bit audio
                    if len(audio_bytes) % 2 != 0:
                        logger.warning(f"⚠️ Padding odd-sized audio chunk: {len(audio_bytes)} bytes")
                        audio_bytes += b'\x00'  # Pad with one zero byte

                    # Convert bytes to numpy array (16-bit PCM)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    logger.info(f"🎵 Processing audio chunk: {len(audio_array)} samples, original base64: {len(msg.audio_data)} chars")

                    # Process audio chunk with the audio relay
                    is_final = getattr(msg, 'is_final', False)
                    await audio_relay.handle_audio_chunk(room, audio_bytes, websocket, is_final)

                except Exception as e:
                    logger.error(f"❌ Error processing audio chunk: {e}")
                    # Send error response
                    error_response = {
                        "type": "audio_error",
                        "error": str(e),
                        "sender_id": "server",
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(error_response))

            else:
                logger.warning(f"⚠️ No audio relay available or no audio data for {client_id}")

        except Exception as e:
            logger.error(f"❌ Error handling audio chunk: {e}")

    async def handle_audio_activity(self, websocket: WebSocket, msg: SignalingMessage):
        """Handle real-time audio activity from client microphone"""
        try:
            client_id = self.websocket_to_id.get(websocket, "unknown")
            room = msg.room or "default"

            logger.info(f"🎤 Received audio activity from {client_id} in room {room}")

            # Import modules for STT processing
            from main import stt_server, call_agent_zero, tts_server
            import base64
            import numpy as np

            if not msg.audio_data:
                logger.warning("No audio data in activity message")
                return

            # Decode base64 audio data
            audio_bytes = base64.b64decode(msg.audio_data)
            # Convert to float32 array for Faster Whisper
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            logger.info(f"🎵 Processing audio activity: {len(audio_array)} samples")

            # Use Faster Whisper STT for transcription
            if stt_server:
                transcript = await stt_server.transcribe_audio_array(audio_array, sample_rate=16000)
                logger.info(f"📝 Transcript: {transcript}")

                if transcript and transcript.strip():
                    # Send transcript back to client immediately
                    transcript_response = {
                        "type": "transcript",
                        "data": {
                            "text": transcript,
                            "isFinal": True
                        },
                        "sender_id": "server",
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(transcript_response))

                    # Get AI response
                    ai_response = await call_agent_zero(transcript)
                    logger.info(f"🤖 AI Response: {ai_response}")

                    # Generate TTS audio response
                    audio_response_base64 = ""
                    if tts_server:
                        try:
                            audio_output = await tts_server.synthesize(ai_response)
                            audio_response_base64 = base64.b64encode(audio_output.tobytes()).decode('utf-8')
                            logger.info(f"🔊 Generated TTS audio: {len(audio_response_base64)} chars")
                        except Exception as tts_error:
                            logger.error(f"TTS Error: {tts_error}")

                    # Send complete response back to client
                    response_msg = {
                        "type": "response",
                        "data": {
                            "transcript": transcript,
                            "text": ai_response,
                            "audio": audio_response_base64,
                            "isTextResponse": False  # This is an audio response
                        },
                        "sender_id": "server",
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(response_msg))

                else:
                    logger.info("No transcript generated or empty transcript")
            else:
                logger.error("STT server not available")
                error_response = {
                    "type": "error",
                    "error": "Speech-to-text service not available",
                    "sender_id": "server",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"❌ Error handling audio activity: {e}")
            logger.error(f"❌ Error details: {str(e)}")
            # Send error response
            error_response = {
                "type": "error",
                "error": f"Audio processing error: {str(e)}",
                "sender_id": "server",
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(error_response))

# Global signaling server instance
signaling_server = WebRTCSignalingServer()
