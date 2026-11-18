"""
Integration Test Suite for Live Webcam Functionality
----------------------------------------------------
Verifies:
  1. GET /api/webcam/status returns inactive when idle
  2. POST /api/webcam/frame returns 400 when no session is active
  3. POST /api/webcam/start initializes session and returns session_id + video_id
  4. POST /api/webcam/frame processes base64 JPEG frame through AI pipeline and returns annotated frame + telemetry
  5. GET /api/webcam/status returns active session metrics
  6. POST /api/webcam/stop finalizes session in SQLite DB and resets pipeline state
  7. Verification that CCTV endpoints (/api/cctv/start, /api/cctv/status) return 404 (completely removed)
"""

from __future__ import annotations

import sys
from pathlib import Path
import base64
import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def create_test_frame_b64(width: int = 640, height: int = 480) -> str:
    """Creates a synthetic test image frame encoded as base64 JPEG."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw simple background shapes
    cv2.rectangle(img, (50, 50), (200, 300), (40, 40, 180), -1)
    cv2.circle(img, (320, 240), 60, (200, 150, 50), -1)
    cv2.putText(img, "TEST WEBCAM FRAME", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_webcam_idle_and_unauthorized_frame():
    """Verifies webcam status when idle and rejects frame processing without active session."""
    # Ensure any lingering session is stopped first
    client.post("/api/webcam/stop")

    res_status = client.get("/api/webcam/status")
    assert res_status.status_code == 200
    assert res_status.json()["active"] is False

    frame_b64 = create_test_frame_b64()
    res_frame = client.post("/api/webcam/frame", json={"image": frame_b64})
    assert res_frame.status_code == 400
    assert "No active webcam session" in res_frame.json()["detail"]


def test_webcam_session_lifecycle_and_frame_processing():
    """Starts session, sends test frame, checks telemetry & annotated image, and stops session."""
    # 1. Start webcam session
    res_start = client.post("/api/webcam/start")
    assert res_start.status_code == 200
    start_data = res_start.json()
    assert start_data["success"] is True
    session_id = start_data["session_id"]
    video_id = start_data["video_id"]
    assert session_id.startswith("ses_cam_")
    assert video_id.startswith("webcam_")

    # Starting again should return the existing active session
    res_start_again = client.post("/api/webcam/start")
    assert res_start_again.status_code == 200
    assert res_start_again.json()["session_id"] == session_id

    # 2. Check active status
    res_status = client.get("/api/webcam/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["active"] is True
    assert status_data["session_id"] == session_id

    # 3. Send test frame
    frame_b64 = create_test_frame_b64()
    res_frame = client.post("/api/webcam/frame", json={"image": frame_b64})
    assert res_frame.status_code == 200
    frame_data = res_frame.json()
    assert frame_data["success"] is True
    assert frame_data["frame_idx"] == 1
    assert "annotated_frame" in frame_data
    assert len(frame_data["annotated_frame"]) > 0
    assert "people" in frame_data
    assert "people_count" in frame_data
    assert "new_incidents" in frame_data

    # 4. Stop session
    res_stop = client.post("/api/webcam/stop")
    assert res_stop.status_code == 200
    stop_data = res_stop.json()
    assert stop_data["success"] is True
    assert stop_data["session_id"] == session_id
    assert stop_data["total_frames_processed"] == 1

    # 5. Status should now be inactive
    res_status_after = client.get("/api/webcam/status")
    assert res_status_after.status_code == 200
    assert res_status_after.json()["active"] is False


def test_cctv_endpoints_removed():
    """Verifies that all CCTV / RTSP endpoints return 404 Not Found."""
    res_cctv_status = client.get("/api/cctv/status")
    assert res_cctv_status.status_code == 404

    res_cctv_start = client.post("/api/cctv/start", json={"rtsp_url": "rtsp://192.168.1.100/stream"})
    assert res_cctv_start.status_code == 404

    res_cctv_stop = client.post("/api/cctv/stop", json={"session_id": "cctv_123"})
    assert res_cctv_stop.status_code == 404

    res_cctv_frame = client.get("/api/cctv/frame?session_id=cctv_123")
    assert res_cctv_frame.status_code == 404
