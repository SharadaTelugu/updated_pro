'''cctv_service.py
Implements server‑side handling of IP CCTV (RTSP) streams.
Provides a `CCTVSession` class that manages a single active RTSP capture,
feeds frames into the existing `VideoBehaviourPipeline`, persists incidents
via `IncidentDatabase`, and makes the latest annotated frame available for
the frontend.
''' 

import threading
import time
import uuid
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any

from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline
from backend.app.services.incident_service import IncidentDatabase, IncidentDetector, EVIDENCE_DIR

class CCTVSession:
    """Manages a live RTSP stream and analysis lifecycle.

    Attributes
    ----------
    session_id: str
        Unique identifier for the analysis session (stored in DB).
    rtsp_url: str
        The original RTSP URL (credentials stripped when logged).
    cap: cv2.VideoCapture
        OpenCV capture object for the stream.
    pipeline: VideoBehaviourPipeline
        Shared pipeline instance (singleton from get_pipeline()).
    incident_detector: IncidentDetector
        Detector instance that writes incidents/evidence.
    thread: threading.Thread
        Background thread reading frames and processing them.
    running: bool
        Flag controlling the read loop.
    latest_frame_bytes: bytes
        JPEG‑encoded annotated frame for quick polling.
    status: str
        One of "idle", "running", "error".
    """

    def __init__(self, rtsp_url: str, pipeline: VideoBehaviourPipeline, incident_db: IncidentDatabase, incident_detector: IncidentDetector):
        self.session_id = f"cctv_{uuid.uuid4().hex[:12]}"
        self.rtsp_url = rtsp_url
        self.pipeline = pipeline
        self.incident_db = incident_db
        self.incident_detector = incident_detector
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.latest_frame_bytes = b""
        self.status = "idle"
        self._create_db_session()

    # ---------------------------------------------------------------------
    # Database helpers
    # ---------------------------------------------------------------------
    def _create_db_session(self) -> None:
        """Create an entry in `analysis_sessions` for this CCTV stream.
        The source type is stored as ``ip_cctv`` and the original URL (masked).
        """
        meta = {
            "source": "ip_cctv",
            "rtsp_url": self._mask_url(self.rtsp_url),
            "started_at": time.time(),
        }
        self.incident_db.create_session(session_id=self.session_id, meta=meta)

    @staticmethod
    def _mask_url(url: str) -> str:
        """Remove credentials from an RTSP URL for safe logging/DB storage.
        Example: ``rtsp://user:pass@host/stream`` -> ``rtsp://host/stream``
        """
        try:
            from urllib.parse import urlparse, urlunparse
            parts = urlparse(url)
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f"{':' + str(parts.port)}"
            masked = parts._replace(netloc=netloc)
            return urlunparse(masked)
        except Exception:
            return url  # fallback – should rarely happen

    # ---------------------------------------------------------------------
    # Lifecycle management
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Open the RTSP stream and launch the processing thread.
        Raises ``HTTPException`` on failure so the API can return a 400/500.
        """
        if self.running:
            raise RuntimeError("CCTV session already running")
        # Use FFMPEG backend for reliable RTSP handling.
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.status = "error"
            raise RuntimeError(f"Unable to open RTSP stream: {self.rtsp_url}")
        self.running = True
        self.status = "running"
        self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"CCTV-{self.session_id}")
        self.thread.start()

    def _read_loop(self) -> None:
        """Continuously read frames, process them and store the latest JPEG.
        The loop terminates when ``self.running`` becomes ``False`` or on
        unrecoverable capture errors.
        """
        retry_count = 0
        max_retries = 3
        while self.running:
            if not self.cap:
                break
            ret, frame = self.cap.read()
            if not ret:
                retry_count += 1
                if retry_count > max_retries:
                    self.status = "error"
                    self.running = False
                    break
                # brief back‑off before retrying
                time.sleep(0.5 * retry_count)
                continue
            retry_count = 0  # reset on success
            # Process the frame – the pipeline returns an annotated BGR array.
            try:
                annotated = self.pipeline.process_frame(frame)
                # Incident detection (behaviour & emotion rules) – uses same API as webcam.
                self.incident_detector.process_frame(
                    frame=annotated,
                    session_id=self.session_id,
                    frame_index=0,  # CCTV does not maintain a per‑frame index for now
                )
                # Encode to JPEG for frontend consumption.
                success, jpeg = cv2.imencode('.jpg', annotated)
                if success:
                    self.latest_frame_bytes = jpeg.tobytes()
            except Exception as exc:
                # Log and continue – a single frame failure should not kill the session.
                print(f"[CCTV {self.session_id}] frame processing error: {exc}")
                continue
        # Cleanup on exit
        self._cleanup()

    def stop(self) -> None:
        """Signal the read loop to exit and wait for the thread to finish."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.status = "idle"
        # Update DB session with stop time and any metrics we gathered.
        self.incident_db.update_session(self.session_id, {"stopped_at": time.time()})

    def _cleanup(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        self.latest_frame_bytes = b""
        self.running = False
        self.status = "idle"

    # ---------------------------------------------------------------------
    # Helper API
    # ---------------------------------------------------------------------
    def get_latest_frame_base64(self) -> str:
        """Return the most recent annotated frame as a base64‑encoded JPEG.
        If no frame is available, returns an empty string.
        """
        if not self.latest_frame_bytes:
            return ""
        return base64.b64encode(self.latest_frame_bytes).decode('utf-8')
