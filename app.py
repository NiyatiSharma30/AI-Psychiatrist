import os
import cv2
import numpy as np
import librosa
import requests
import json
import speech_recognition as sr
from fastapi import FastAPI, WebSocket
import base64
from deepface import DeepFace
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
import av
import asyncio

app = FastAPI()
pcs = set()
recognizer = sr.Recognizer()

DID_API_KEY = "bml5YXRpc2hhcm1hMzAxMUBnbWFpbC5jb20:mVw8viI7BxfSiB3D2Xx_w"
DID_URL = "https://api.d-id.com/talks"
OPENAI_API_KEY = "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"


@app.websocket("/emotion")
async def emotion_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            jpg_as_np = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotions = result[0]["dominant_emotion"] if result else "unknown"
            await websocket.send_json({"emotion": emotions})
        except Exception as e:
            print(e)
            break

class AudioEmotionTrack(AudioStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        audio = frame.to_ndarray()
        y = audio.astype(np.float32)
        sr = 44100  
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]  
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        energy = np.sum(y ** 2) / len(y)
        return av.AudioFrame.from_ndarray(audio, layout="mono")

@app.post("/offer")
async def offer(offer: dict):
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
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    prompt = f"The user seems {emotion}. Respond with empathy."

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

@app.post("/generate_avatar/")
async def generate_avatar(emotion: str):
    ai_response = generate_ai_response(emotion)
    headers = {"Authorization": f"Bearer {DID_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "script": {"type": "text", "input": ai_response},
        "source_url": "https://your-avatar-url.com",
        "driver_url": "bank://lively"
    }
    response = requests.post(DID_URL, headers=headers, json=payload)
    return response.json()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
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
        except Exception as e:
            print(f"WebSocket error: {e}")
            break 

@app.get("/analyze")
async def analyze_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error": "Could not capture frame"}
    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    return {"emotion": result[0]["dominant_emotion"]} if result else {"emotion": "unknown"}

@app.get("/")
async def root():
    return {"message": "AI Psychiatrist API is running!"}


