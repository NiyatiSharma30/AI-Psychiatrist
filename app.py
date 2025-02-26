import os
import cv2
import numpy as np
import librosa
import requests
import json
import speech_recognition as sr
import base64
import asyncio
import av

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack

DID_API_KEY = "bml5YXRpc2hhcm1hMzAxMUBnbWFpbC5jb20:mVw8viI7BxfSiB3D2Xx_w"
DID_URL = "https://api.d-id.com/talks"
OPENAI_API_KEY = "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"

app = FastAPI()
pcs = set()
recognizer = sr.Recognizer()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fix_base64_padding(data):
    """Ensures Base64 string has correct padding."""
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return data


@app.websocket("/emotion")
async def emotion_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time emotion detection."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data = fix_base64_padding(data)

            try:
                decoded_data = base64.b64decode(data)
            except Exception as e:
                print(f"❌ Base64 Decode Error: {e}")
                await websocket.send_json({"error": "Invalid Base64 data"})
                continue

            jpg_as_np = np.frombuffer(decoded_data, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                await websocket.send_json({"error": "Invalid image"})
                continue

            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                emotion = result[0]["dominant_emotion"] if result else "unknown"
            except Exception as e:
                print(f"❌ DeepFace Error: {e}")
                emotion = "unknown"

            await websocket.send_json({"emotion": emotion})

    except WebSocketDisconnect:
        print("❌ WebSocket Disconnected")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
    finally:
        await websocket.close()


class AudioEmotionTrack(AudioStreamTrack):
    """Analyzes emotion from audio pitch and energy levels."""
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.sr = 44100  # Sample rate

    async def recv(self):
        frame = await self.track.recv()
        audio = frame.to_ndarray()
        y = audio.astype(np.float32)

        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        pitch_values = pitches[pitches > 0]
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        energy = np.sum(y ** 2) / len(y)

        return av.AudioFrame.from_ndarray(audio, layout="mono")


@app.post("/offer")
async def offer(offer: dict):
    """Handles WebRTC offer and returns an answer."""
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoStreamTrack(track))
        elif track.kind == "audio":
            pc.addTrack(AudioEmotionTrack(track))

    await pc.setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def generate_ai_response(emotion):
    """Uses OpenAI API to generate an AI therapist response based on detected emotion."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    prompt = f"The user seems {emotion}. Respond with empathy."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a compassionate AI psychiatrist."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "I'm here to help you."


@app.post("/generate_avatar/")
async def generate_avatar(emotion: str):
    """Generates an avatar video response using D-ID API based on detected emotion."""
    ai_response = generate_ai_response(emotion)
    headers = {"Authorization": f"Bearer {DID_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "script": {"type": "text", "input": ai_response},
        "source_url": "https://myhost.com/image.jpg",
        "driver_url": "bank://lively/driver-05",
    }

    try:
        response = requests.post("https://api.d-id.com/talks", headers=headers, json=payload)
        avatar_data = response.json()
        return {"result_url": avatar_data.get("result_url", "No Video")}
    except Exception as e:
        print(f"❌ D-ID API Error: {e}")
        return {"error": "Avatar generation failed"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time emotion and avatar video streaming."""
    await websocket.accept()
    try:
        while True:
            emotion = "neutral"
            pitch = 0.0
            energy = 0.0

            avatar_response = await generate_avatar(emotion)
            await websocket.send_json({
                "emotion": emotion,
                "pitch": pitch,
                "energy": energy,
                "avatar_video_url": avatar_response.get("result_url", "No Video")
            })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("❌ WebSocket Disconnected")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
    finally:
        await websocket.close()


@app.get("/analyze")
async def analyze_emotion():
    """Captures a single frame and analyzes emotion using DeepFace."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Could not capture frame"}

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        return {"emotion": result[0]["dominant_emotion"]} if result else {"emotion": "unknown"}
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")
        return {"emotion": "unknown"}


@app.get("/")
async def root():
    """Root endpoint for API status check."""
    return {"message": "AI Psychiatrist API is running!"}
