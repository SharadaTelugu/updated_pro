import pytest
import builtins
import json
from unittest.mock import patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
sys.path.append('C:/Users/naroj/OneDrive/Desktop/prjct2')
from backend.app.main import app, incident_db

client = TestClient(app)

# Helper mock for cv2.VideoCapture
class MockVideoCapture:
    def __init__(self, url, *args, **kwargs):
        self.url = url
        # Decide behavior based on URL suffix
        self._should_open = not url.endswith('fail')
        self._read_success = not url.endswith('noread')
        self.released = False

    def isOpened(self):
        return self._should_open

    def read(self):
        if not self._read_success:
            return (False, None)
        # Return a dummy black frame (BGR)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return (True, frame)

    def release(self):
        self.released = True

# Mock for socket.gethostbyname to map any hostname to a private IP
def mock_gethostbyname(hostname):
    # Simple mapping: if hostname contains "invalid" raise error
    if 'invalid' in hostname:
        raise OSError('host not found')
    # Return a private IP for all other hostnames
    return '192.168.1.10'

# Fixture to clear the CCTV session registry before each test
@pytest.fixture(autouse=True)
def clear_cctv_sessions():
    # Access the internal registry via the app module
    from backend.app.main import _cctv_sessions, _cctv_lock
    with _cctv_lock:
        _cctv_sessions.clear()
    yield
    # Ensure any leftover sessions are stopped
    with _cctv_lock:
        for entry in list(_cctv_sessions.values()):
            session = entry["session"]
            try:
                session.stop()
            except Exception:
                pass
        _cctv_sessions.clear()

# ---- URL validation tests ---------------------------------------------------
@pytest.mark.parametrize('url,expected_status,detail_substring', [
    ('rtsp://camera.local/stream', 200, None),                    # valid
    ('http://camera.local/stream', 400, 'rtsp scheme'),           # wrong scheme
    ('rtsp:///stream', 400, 'hostname'),                          # missing host
    ('rtsp://invalid.local/stream', 400, 'resolve'),             # DNS fail
    ('rtsp://127.0.0.1/stream', 403, 'private network'),          # loopback
    ('rtsp://8.8.8.8/stream', 403, 'private network'),            # public IP
])
def test_rtsp_url_validation(url, expected_status, detail_substring):
    with patch('socket.gethostbyname', side_effect=mock_gethostbyname):
        response = client.post('/api/cctv/start', json={'rtsp_url': url})
    assert response.status_code == expected_status
    if detail_substring:
        assert detail_substring in response.text.lower()

# ---- Successful start and frame retrieval -----------------------------------
@patch('cv2.VideoCapture', MockVideoCapture)
@patch('socket.gethostbyname', side_effect=mock_gethostbyname)
def test_start_success_and_frame(mock_gethost):
    # Start session
    resp = client.post('/api/cctv/start', json={'rtsp_url': 'rtsp://camera.local/stream'})
    assert resp.status_code == 200
    data = resp.json()
    session_id = data['session_id']
    assert data['status'] == 'running'

    # Retrieve frame – first call may be 204 because background thread hasn't produced yet
    frame_resp = client.get('/api/cctv/frame', params={'session_id': session_id})
    # Accept either 204 (no frame yet) or 200 with a base64 string
    if frame_resp.status_code == 200:
        payload = frame_resp.json()
        assert payload['session_id'] == session_id
        assert isinstance(payload['frame_base64'], str)
        assert len(payload['frame_base64']) > 0
    else:
        assert frame_resp.status_code == 204

    # Stop session
    stop_resp = client.post('/api/cctv/stop', json={'session_id': session_id})
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data['status'] == 'idle'

# ---- Capture opening failure ------------------------------------------------
@patch('cv2.VideoCapture', MockVideoCapture)
@patch('socket.gethostbyname', side_effect=mock_gethostbyname)
def test_start_failure_due_to_capture(mock_gethost):
    # URL ending with 'fail' triggers MockVideoCapture.isOpened() == False
    resp = client.post('/api/cctv/start', json={'rtsp_url': 'rtsp://camera.local/fail'})
    assert resp.status_code == 500
    assert 'Failed to open RTSP stream' in resp.text

# ---- Credential masking verification ---------------------------------------
@patch('cv2.VideoCapture', MockVideoCapture)
@patch('socket.gethostbyname', side_effect=mock_gethostbyname)
def test_url_masking_in_db(mock_gethost):
    url = 'rtsp://user:secret@camera.local/stream'
    resp = client.post('/api/cctv/start', json={'rtsp_url': url})
    assert resp.status_code == 200
    session_id = resp.json()['session_id']
    # Query the underlying SQLite DB directly via IncidentDatabase helper
    meta = incident_db.get_session_meta(session_id)
    # The stored URL should have credentials stripped
    assert 'user' not in meta['rtsp_url']
    assert 'secret' not in meta['rtsp_url']
    assert meta['rtsp_url'].startswith('rtsp://camera.local')

# ---- Multiple concurrent sessions ------------------------------------------
@patch('cv2.VideoCapture', MockVideoCapture)
@patch('socket.gethostbyname', side_effect=mock_gethostbyname)
def test_multiple_sessions(mock_gethost):
    ids = []
    for i in range(3):
        resp = client.post('/api/cctv/start', json={'rtsp_url': f'rtsp://camera{i}.local/stream'})
        assert resp.status_code == 200
        ids.append(resp.json()['session_id'])
    # Verify all sessions are registered
    from backend.app.main import _cctv_sessions, _cctv_lock
    with _cctv_lock:
        assert len(_cctv_sessions) == 3
    # Stop the middle session
    stop_resp = client.post('/api/cctv/stop', json={'session_id': ids[1]})
    assert stop_resp.status_code == 200
    with _cctv_lock:
        assert len(_cctv_sessions) == 2
        assert ids[1] not in _cctv_sessions
    # Remaining sessions should still be running
    for sid in (ids[0], ids[2]):
        status_resp = client.get('/api/cctv/status')
        assert any(s['session_id'] == sid and s['status'] == 'running' for s in status_resp.json()['active_sessions'])

# ---- Ensure no credentials appear in logs (simple sanity check) ----------
@patch('cv2.VideoCapture', MockVideoCapture)
@patch('socket.gethostbyname', side_effect=mock_gethostbyname)
def test_no_credentials_in_logs(caplog, mock_gethost):
    url = 'rtsp://admin:pwd@camera.local/stream'
    client.post('/api/cctv/start', json={'rtsp_url': url})
    # Check captured logs for the raw URL – should not contain credentials
    for record in caplog.records:
        assert 'admin' not in record.getMessage()
        assert 'pwd' not in record.getMessage()
