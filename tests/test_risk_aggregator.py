"""Tests for RiskAggregator - Phase 2 integration"""
import pytest
from pathlib import Path

# Import using relative path from project root
from backend.app.services.risk_aggregator import RiskAggregator


def test_behaviour_emotion_weighting():
    agg = RiskAggregator(smoothing_window=5)
    records = [
        {
            "track_id": 1,
            "behaviour": {"probabilities": {"Fight": 0.5, "Fall": 0.2, "NonFight": 0.3}},
            "emotion": {"probabilities": {"Angry": 0.6, "Happy": 0.4}, "status": "available"},
        }
    ]
    enriched = agg.aggregate(records)
    rec = enriched[0]
    # Behaviour contribution = 0.7 * (0.5+0.2) = 0.49
    assert pytest.approx(rec["behaviour_score"], rel=1e-4) == 0.49
    # Raw emotion score = (0.6*1.0 + 0.4*0.0) = 0.6; weighted = 0.6 * 0.3 = 0.18
    assert pytest.approx(rec["emotion_score"], rel=1e-4) == 0.18
    # Final risk = sum = 0.67
    assert pytest.approx(rec["final_risk_score"], rel=1e-4) == 0.67
    # Contributions reported correctly
    assert rec["behaviour_contribution"] == 0.7
    assert rec["emotion_contribution"] == 0.3


def test_missing_face_handling():
    agg = RiskAggregator()
    records = [
        {"track_id": 2, "behaviour": {"probabilities": {"Fight": 0.1, "Fall": 0.0}}, "emotion": {"status": "unavailable"}}
    ]
    enriched = agg.aggregate(records)
    rec = enriched[0]
    # Emotion score should be None and contribution 0
    assert rec["emotion_score"] is None
    assert rec["emotion_contribution"] == 0.0
    # Final risk equals behaviour contribution only
    expected_behaviour = 0.7 * (0.1 + 0.0)
    assert pytest.approx(rec["final_risk_score"], rel=1e-4) == expected_behaviour


def test_emotion_smoothing():
    agg = RiskAggregator(smoothing_window=3)
    # First frame raw emotion score 0.2 (e.g., low risk)
    rec1 = {"track_id": 3, "behaviour": {"probabilities": {"Fight": 0.0, "Fall": 0.0}}, "emotion": {"probabilities": {"Sad": 0.5}, "status": "available"}}
    # Sad maps to 0.4 risk factor -> raw = 0.5*0.4=0.2, weighted = 0.2*0.3=0.06
    agg.aggregate([rec1])
    first_score = rec1["emotion_score"]
    # Second frame higher risk emotion Angry 0.8
    rec2 = {"track_id": 3, "behaviour": {"probabilities": {"Fight": 0.0, "Fall": 0.0}}, "emotion": {"probabilities": {"Angry": 0.8}, "status": "available"}}
    agg.aggregate([rec2])
    second_score = rec2["emotion_score"]
    # Smoothed score after two frames = (0.06 + (0.8*1.0*0.3=0.24)) / 2 = 0.15
    assert pytest.approx(second_score, rel=1e-4) == (first_score + 0.24) / 2


def test_multiple_tracks_separate_history():
    agg = RiskAggregator(smoothing_window=2)
    rec_a1 = {"track_id": 10, "behaviour": {"probabilities": {}}, "emotion": {"probabilities": {"Angry": 0.9}, "status": "available"}}
    rec_b1 = {"track_id": 20, "behaviour": {"probabilities": {}}, "emotion": {"probabilities": {"Sad": 0.9}, "status": "available"}}
    agg.aggregate([rec_a1, rec_b1])
    # Second frame only track 10 updates, track 20 should retain its previous history
    rec_a2 = {"track_id": 10, "behaviour": {"probabilities": {}}, "emotion": {"probabilities": {"Angry": 0.1}, "status": "available"}}
    agg.aggregate([rec_a2])
    # Smoothed for track 10 = avg(0.9*1*0.3=0.27 , 0.1*1*0.3=0.03) = 0.15
    assert pytest.approx(rec_a2["emotion_score"], rel=1e-4) == (0.27 + 0.03) / 2
    # Track 20's last emotion_score should still be its first smoothed value (0.9*0.4*0.3=0.108)
    # Retrieve by re‑aggregating original rec_b1 to inspect stored history
    agg.aggregate([rec_b1])
    expected_b = 0.9 * 0.4 * 0.3  # Sad risk factor 0.4
    assert pytest.approx(rec_b1["emotion_score"], rel=1e-4) == expected_b
