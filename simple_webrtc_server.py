#!/usr/bin/env python3
"""
Simple WebRTC Signaling Server
Minimal server for testing WebRTC voice connections without voice service dependencies
"""

import asyncio
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple signaling server
rooms = {}
websocket_to_room = {}
websocket_to_id = {}
next_id = 0

def generate_id():
    global next_id
    next_id += 1
    return f"client_{next_id}"

# Create FastAPI app
app = FastAPI(title="WebRTC Signaling Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "signaling_available": True,
        "timestamp": time.time()
    }

@app.get("/fastrtc/status")
async def signaling_status():
    return {
        "signaling_available": True,
        "websocket_endpoint": "/ws",
        "transport": "webrtc_signaling",
        "timestamp": time.time()
    }

@app.websocket("/ws")
async def websocket_signaling(websocket: WebSocket):
    """WebSocket endpoint for WebRTC signaling"""
    await websocket.accept()
    client_id = generate_id()
    websocket_to_id[websocket] = client_id

    logger.info(f"✅ WebRTC signaling connection: {client_id}")

    # Send welcome message
    await websocket.send_text(json.dumps({
        "type": "welcome",
        "client_id": client_id,
        "timestamp": time.time()
    }))

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")

            logger.info(f"📨 Message: {msg_type} from {client_id}")

            if msg_type == "join":
                room = data.get("room", "default")

                # Leave previous room
                if websocket in websocket_to_room:
                    old_room = websocket_to_room[websocket]
                    rooms[old_room].discard(websocket)
                    if not rooms[old_room]:
                        del rooms[old_room]

                # Join new room
                if room not in rooms:
                    rooms[room] = set()
                rooms[room].add(websocket)
                websocket_to_room[websocket] = room

                logger.info(f"👥 {client_id} joined room: {room}")

                # Send joined confirmation
                await websocket.send_text(json.dumps({
                    "type": "joined",
                    "room": room,
                    "client_id": client_id,
                    "room_size": len(rooms[room]),
                    "timestamp": time.time()
                }))

                # Notify others in room
                for peer in rooms[room]:
                    if peer != websocket:
                        try:
                            await peer.send_text(json.dumps({
                                "type": "peer_joined",
                                "client_id": client_id,
                                "timestamp": time.time()
                            }))
                        except:
                            pass

            elif msg_type in ["offer", "answer", "ice"]:
                # Relay WebRTC signaling messages
                if websocket in websocket_to_room:
                    room = websocket_to_room[websocket]
                    data["sender_id"] = client_id
                    data["timestamp"] = time.time()
                    message_json = json.dumps(data)

                    # Send to all other peers in room
                    for peer in rooms.get(room, []):
                        if peer != websocket:
                            try:
                                await peer.send_text(message_json)
                                logger.info(f"🔄 Relayed {msg_type} to peer")
                            except:
                                pass

            elif msg_type == "data":
                # Handle data channel messages (for voice processing)
                logger.info(f"📊 Data message from {client_id}")

                # Simple echo response for testing
                response = {
                    "type": "data_response",
                    "data": {
                        "type": "response",
                        "text": "WebRTC signaling server received your message",
                        "original": data.get("data", {})
                    },
                    "sender_id": "server",
                    "timestamp": time.time()
                }

                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info(f"🔌 {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        # Cleanup
        if websocket in websocket_to_room:
            room = websocket_to_room[websocket]
            rooms[room].discard(websocket)
            if not rooms[room]:
                del rooms[room]
            else:
                # Notify others
                for peer in rooms[room]:
                    try:
                        await peer.send_text(json.dumps({
                            "type": "peer_left",
                            "client_id": client_id,
                            "timestamp": time.time()
                        }))
                    except:
                        pass
            del websocket_to_room[websocket]

        websocket_to_id.pop(websocket, None)

if __name__ == "__main__":
    print("Starting Simple WebRTC Signaling Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)