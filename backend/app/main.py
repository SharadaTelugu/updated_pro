"""
Student Safety Monitoring System - Backend API Server
------------------------------------------------------
FastAPI server exposing endpoints to run the Unified Multi-Branch Video Pipeline
and serve the Frontend Dashboard.
"""

from __future__ import annotations

import sys
import os
import shutil
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional

import threading
import ipaddress
import socket
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------
# Project root handling
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------
# Service imports (project specific)
# ---------------------------------------------------------
from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline
from backend.app.services.incident_service import IncidentDatabase, IncidentDetector, EVIDENCE_DIR, DB_PATH
from backend.app.services.cctv_service import CCTVSession

# ---------------------------------------------------------
# Directories used by the API
# ---------------------------------------------------------
STORAGE_DIR = ROOT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
OUTPUT_DIR = STORAGE_DIR / "outputs"
FRONTEND_DIR = ROOT_DIR / "frontend"
ASSETS_DIR = ROOT_DIR / "tests" / "assets"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Incident database singletons (stage 9C)
# ---------------------------------------------------------
incident_db = IncidentDatabase(DB_PATH)
incident_detector = IncidentDetector(db=incident_db, evidence_dir=EVIDENCE_DIR)

# ---------------------------------------------------------
# Pipeline singletons
# ---------------------------------------------------------
_pipeline_instance: Optional[VideoBehaviourPipeline] = None

def get_pipeline() -> VideoBehaviourPipeline:
    """Return a cached singleton pipeline instance to avoid re‑loading models."""
    global _pipeline_instance
    if _pipeline_instance is None:
        print("[API] Initializing Unified Multi‑Branch Video Pipeline singleton...")
        _pipeline_instance = VideoBehaviourPipeline()
        print("[API] Pipeline singleton ready.")
    return _pipeline_instance

# Dedicated webcam pipeline (isolated from batch analysis)
_webcam_pipeline_instance: Optional[VideoBehaviourPipeline] = None
_webcam_pipeline_lock = threading.Lock()

def get_webcam_pipeline() -> VideoBehaviourPipeline:
    """Return a dedicated webcam pipeline instance."""
    global _webcam_pipeline_instance
    if _webcam_pipeline_instance is None:
        print("[API] Initializing dedicated Webcam Pipeline instance...")
        _webcam_pipeline_instance = VideoBehaviourPipeline()
        print("[API] Webcam Pipeline ready.")
    return _webcam_pipeline_instance

# Active webcam session state (single session allowed)
_webcam_session: Dict[str, Any] = {
    "active": False,
    "session_id": None,
}

# ---------------------------------------------------------
# FastAPI application (single instance)
# ---------------------------------------------------------
app = FastAPI(
    title="Student Safety Monitoring System",
    description="Multi‑Branch AI System for Student Safety & Behaviour Analysis",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# CCTV request models and helper utilities
# ---------------------------------------------------------
class CCTVStartRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL of the IP camera (e.g., rtsp://host/stream)")
    camera_name: Optional[str] = Field(None, description="Optional friendly name for the camera")

class CCTVStopRequest(BaseModel):
    session_id: str = Field(..., description="CCTV session identifier to stop")

def _mask_url(url: str) -> str:
    """Remove credentials from an RTSP URL for safe logging/storage."""
    try:
        parts = urlparse(url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        return urlunparse((parts.scheme, netloc, parts.path, parts.params, parts.query, parts.fragment))
    except Exception:
        return url

def _is_allowed_ip(host: str) -> bool:
    """Allow only private network IPs (no loopback or public)."""
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private and not ip.is_loopback
    except ValueError:
        return False

def _validate_rtsp_url(rtsp_url: str) -> str:
    """Validate RTSP URL scheme, resolve hostname, and enforce private network restriction."""
    parts = urlparse(rtsp_url)
    if parts.scheme.lower() != "rtsp":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL must use rtsp scheme")
    host = parts.hostname
    if not host:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="RTSP URL must contain a hostname")
    try:
        ip = socket.gethostbyname(host)
    except socket.gaierror:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to resolve RTSP hostname")
    if not _is_allowed_ip(ip):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="RTSP host is not in an allowed private network")
    return rtsp_url

# ---------------------------------------------------------
# Global CCTV session registry (single active session for now)
# ---------------------------------------------------------
_cctv_sessions: Dict[str, CCTVSession] = {}
_cctv_lock = threading.Lock()

# ---------------------------------------------------------
# CCTV API Endpoints
# ---------------------------------------------------------
@app.post("/api/cctv/start")
def cctv_start(request: CCTVStartRequest):
    """Start a new CCTV RTSP session."""
    rtsp_url = _validate_rtsp_url(request.rtsp_url)
    masked = _mask_url(rtsp_url)
    print(f"[API] Starting CCTV session for URL: {masked}")

    pipeline = get_pipeline()
    cam_incident_detector = IncidentDetector(db=incident_db, evidence_dir=EVIDENCE_DIR)
    try:
        session = CCTVSession(rtsp_url=rtsp_url, pipeline=pipeline, incident_db=incident_db, incident_detector=cam_incident_detector)
        session.start()
    except Exception as e:
        print(f"[API] Failed to start CCTV session: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to open RTSP stream")

    entry = {"session": session, "camera_name": request.camera_name, "started_at": time.time()}
    with _cctv_lock:
        _cctv_sessions[session.session_id] = entry
    return {"success": True, "session_id": session.session_id, "status": session.status, "camera_name": request.camera_name}

@app.post("/api/cctv/stop")
def cctv_stop(body: CCTVStopRequest):
    """Stop an active CCTV session."""
    with _cctv_lock:
        entry = _cctv_sessions.get(body.session_id)
        if not entry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CCTV session not found")
        session: CCTVSession = entry["session"]
        del _cctv_sessions[body.session_id]
    try:
        session.stop()
    except Exception as e:
        print(f"[API] Error stopping CCTV session {body.session_id}: {e}")
    return {"success": True, "session_id": body.session_id, "status": session.status}

@app.get("/api/cctv/status")
def cctv_status():
    """Return status of all active CCTV sessions."""
    now = time.time()
    with _cctv_lock:
        statuses = []
        for sid, entry in _cctv_sessions.items():
            session: CCTVSession = entry["session"]
            elapsed = round(now - entry["started_at"], 2)
            statuses.append({
                "session_id": sid,
                "camera_name": entry.get("camera_name"),
                "status": session.status,
                "elapsed_seconds": elapsed,
                "has_frame": bool(session.latest_frame_bytes),
            })
    return {"active_sessions": statuses}

@app.get("/api/cctv/frame")
def cctv_frame(session_id: str = Query(..., description="CCTV session identifier")):
    """Retrieve latest annotated frame for a session."""
    with _cctv_lock:
        entry = _cctv_sessions.get(session_id)
        if not entry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CCTV session not found")
        session: CCTVSession = entry["session"]
    frame_b64 = session.get_latest_frame_base64()
    if not frame_b64:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="No frame available yet")
    return {"session_id": session_id, "frame_base64": frame_b64}

# ---------------------------------------------------------
# Video upload & analysis utilities
# ---------------------------------------------------------
video_registry: Dict[str, Dict[str, Any]] = {}
analysis_progress: Dict[str, Dict[str, Any]] = {}
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.get("/api/progress/{video_id}")
def get_analysis_progress(video_id: str):
    """Return real‑time progress for an ongoing video analysis task."""
    if video_id in analysis_progress:
        return analysis_progress[video_id]
    return {
        "status": "idle",
        "current_frame": 0,
        "total_frames": 0,
        "percent": 0.0,
        "fps": 0.0,
    }

@app.get("/api/health")
def health_check():
    """Return health status, active models, and pipeline state."""
    pipeline_ready = _pipeline_instance is not None
    return {
        "status": "healthy",
        "system": "Student Safety Monitoring System",
        "stage": "Stage 9C - Complete Application",
        "pipeline_initialized": pipeline_ready,
        "models": {
            "yolo11": "yolo11n.pt (Person Class 0)",
            "pose": "pose_landmarker_lite.task (33 3D Keypoints)",
            "behaviour": "behaviour_model_rich_165_v2_best.keras (BiLSTM, 165 Features)",
            "face": "blaze_face_short_range.tflite (MediaPipe FaceDetector)",
            "emotion": "rafdb_emotion_best.keras (EfficientNetB0, 7 Classes)",
        },
        "supported_formats": list(SUPPORTED_EXTENSIONS),
    }

@app.get("/api/sample-videos")
def list_sample_videos():
    """List pre‑existing sample videos in tests/assets for easy testing."""
    samples = []
    if ASSETS_DIR.exists():
        for p in ASSETS_DIR.glob("*.mp4"):
            if "output" in p.name.lower():
                continue
            cap = cv2.VideoCapture(str(p))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            cap.release()
            duration = round(fc / fps, 2) if fps > 0 else 0.0
            samples.append({
                "filename": p.name,
                "file_path": str(p),
                "file_size": p.stat().st_size,
                "width": w,
                "height": h,
                "frames": fc,
                "fps": round(fps, 1),
                "duration_sec": duration,
            })
    return {"samples": samples}

@app.post("/api/use-sample/{sample_name}")
def use_sample_video(sample_name: str):
    """Copy a sample video from tests/assets into upload storage for analysis."""
    sample_path = ASSETS_DIR / sample_name
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample video '{sample_name}' not found in test assets.",
        )
    video_id = uuid.uuid4().hex[:12]
    dest_path = UPLOAD_DIR / f"{video_id}_{sample_path.name}"
    shutil.copyfile(str(sample_path), str(dest_path))
    cap = cv2.VideoCapture(str(dest_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    meta = {
        "video_id": video_id,
        "filename": sample_path.name,
        "file_path": str(dest_path),
        "file_size": dest_path.stat().st_size,
        "width": w,
        "height": h,
        "frames": fc,
        "fps": round(fps, 2),
        "duration_sec": round(fc / fps, 2) if fps > 0 else 0.0,
        "uploaded_at": time.time(),
        "preview_url": f"/api/video/{video_id}/input",
    }
    video_registry[video_id] = meta
    return meta

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept an uploaded video file, validate it, and store for later analysis."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided in upload.")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    video_id = uuid.uuid4().hex[:12]
    safe_name = f"{video_id}_{file.filename}"
    dest_path = UPLOAD_DIR / safe_name
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    # Metadata extraction could be added here if needed
    return {"video_id": video_id, "filename": file.filename, "saved_path": str(dest_path)}
