import pytest
import os, sys
# Ensure the project root is on PYTHONPATH for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.services.video_behaviour_pipeline import VideoBehaviourPipeline

# The heavy models (YOLO, MediaPipe Pose, BiLSTM, BlazeFace, RAF‑DB) are loaded
# inside the pipeline constructor.  Creating a new pipeline for every test would
# be prohibitively slow and would repeatedly allocate GPU/CPU resources.
# Therefore we expose a **session‑scoped** fixture that builds the pipeline once
# and re‑uses it across all tests that request it.

@pytest.fixture(scope="session")
def pipeline() -> VideoBehaviourPipeline:
    """Return a shared :class:`VideoBehaviourPipeline` instance.

    The fixture is deliberately *session* scoped because the pipeline caches the
    loaded models internally.  Tests that mutate internal state (e.g., replace
    ``association_service.associate_frame``) are expected to restore the original
    callable after the test, so sharing the instance does not cause cross‑test
    contamination.
    """
    pipe = VideoBehaviourPipeline()
    yield pipe
    # No explicit shutdown method exists; rely on garbage collection.
