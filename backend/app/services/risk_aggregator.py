'''Risk Aggregator Service

Combines behaviour AI (YOLO + Pose + BiLSTM) with facial AI (BlazeFace + RAF‑DB) to produce a unified risk score per person.

- Behaviour contribution weight: 0.7 (70%)
- Emotion contribution weight: 0.3 (30%)
'''  # noqa: D400

from collections import defaultdict, deque
from typing import List, Dict, Any, Optional


class RiskAggregator:
    """Calculate a combined risk assessment for each tracked person.

    The aggregator maintains a short history of emotion scores per ``track_id`` to smooth
    frame‑to‑frame fluctuations. Missing or unavailable face data is handled gracefully –
    the emotion contribution is omitted and does not penalise the final risk score.
    """

    BEHAVIOUR_WEIGHT: float = 0.7  # 70 % weight
    EMOTION_WEIGHT: float = 0.3   # 30 % weight

    # Simple mapping of emotion classes to a risk multiplier (0‑1). High‑risk emotions
    # increase the overall score, low‑risk emotions have a minimal impact. This mapping is
    # *illustrative* and not claimed to be scientifically validated.
    EMOTION_RISK_MAP = {
        "Angry": 1.0,
        "Disgust": 0.8,
        "Fear": 0.7,
        "Surprise": 0.6,
        "Sad": 0.4,
        "Neutral": 0.2,
        "Happy": 0.0,
    }

    def __init__(self, smoothing_window: int = 5) -> None:
        self.smoothing_window = smoothing_window
        # Per‑track deque storing recent emotion scores for smoothing
        self._emotion_history: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _behaviour_score(self, behaviour: Dict[str, Any]) -> float:
        """Calculate the behaviour component of the risk.

        Uses the *probabilities* of the ``Fight`` and ``Fall`` classes. The sum of those
        probabilities is multiplied by the configured behaviour weight.
        """
        probs = behaviour.get("probabilities", {})
        fight = float(probs.get("Fight", 0))
        fall = float(probs.get("Fall", 0))
        raw = fight + fall
        return raw * self.BEHAVIOUR_WEIGHT

    def _raw_emotion_score(self, emotion: Dict[str, Any]) -> Optional[float]:
        """Return a raw (unsmoothed) emotion risk score.

        ``emotion`` is expected to contain a ``probabilities`` mapping from emotion class to
        confidence. Each class is multiplied by a handcrafted risk factor defined in
        ``EMOTION_RISK_MAP``. The weighted sum is then scaled by the overall emotion weight.

        Returns ``None`` when the emotion data is unavailable.
        """
        if emotion.get("status") == "unavailable":
            return None
        probs = emotion.get("probabilities", {})
        if not probs:
            return None
        weighted_sum = 0.0
        for label, prob in probs.items():
            factor = self.EMOTION_RISK_MAP.get(label, 0.0)
            weighted_sum += float(prob) * factor
        return weighted_sum * self.EMOTION_WEIGHT

    def _smoothed_emotion_score(self, track_id: int, raw_score: Optional[float]) -> Optional[float]:
        """Smooth the emotion score over the recent ``smoothing_window`` frames.

        If ``raw_score`` is ``None`` the history for the track is *not* updated and the
        function returns ``None`` – this preserves the "missing" status.
        """
        if raw_score is None:
            return None
        hist = self._emotion_history[track_id]
        hist.append(raw_score)
        return sum(hist) / len(hist)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def aggregate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich ``records`` with combined risk information.

        ``records`` is the list returned by ``FacePersonAssociationService`` – each entry
        contains ``track_id``, ``behaviour`` and ``emotion`` keys. The method mutates each
        record in‑place and also returns the enriched list for convenience.
        """
        enriched: List[Dict[str, Any]] = []
        for rec in records:
            tid = rec.get("track_id")
            behaviour = rec.get("behaviour", {})
            emotion = rec.get("emotion", {})

            # Behaviour component (always present)
            behaviour_score = self._behaviour_score(behaviour)

            # Emotion component – may be unavailable
            raw_emotion = self._raw_emotion_score(emotion)
            emotion_score = self._smoothed_emotion_score(tid, raw_emotion)

            emotion_contribution = self.EMOTION_WEIGHT if emotion_score is not None else 0.0
            final_score = behaviour_score + (emotion_score or 0.0)

            # Simple risk level thresholds (illustrative)
            if final_score < 0.30:
                risk_level = "Low"
            elif final_score < 0.60:
                risk_level = "Medium"
            else:
                risk_level = "High"

            modality_status = "available" if emotion_score is not None else "unavailable"

            # Attach fields required by the specification
            rec.update({
                "behaviour_score": behaviour_score,
                "emotion_score": emotion_score,
                "behaviour_contribution": self.BEHAVIOUR_WEIGHT,
                "emotion_contribution": emotion_contribution,
                "final_risk_score": final_score,
                "risk_level": risk_level,
                "modality_status": modality_status,
            })
            enriched.append(rec)
        return enriched
