
from collections.abc import Iterable

from minesweeper.recognition.detection import GridDetection
from minesweeper.session import GameSession


def intersection_over_union(
    a: GridDetection,
    b: GridDetection,
) -> float:
    """Return bounding-box overlap in the range [0, 1]."""

    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)

    width = max(0, right - left)
    height = max(0, bottom - top)

    intersection = width * height
    union = (
        a.width * a.height
        + b.width * b.height
        - intersection
    )

    return intersection / union if union > 0 else 0.0


class GameManager:
    def __init__(
        self,
        match_threshold: float = 0.3,
        max_missed_scans: int = 2,
    ):
        if not 0 <= match_threshold <= 1:
            raise ValueError("Invalid match threshold")

        if max_missed_scans < 1:
            raise ValueError("Invalid missed-scan limit")

        self.match_threshold = match_threshold
        self.max_missed_scans = max_missed_scans

        self.sessions: list[GameSession] = []
        self.next_game_number = 1

    def update(
        self,
        detections: Iterable[GridDetection],
    ) -> None:
        detections = list(detections)

        # Find all plausible session/detection pairs.
        candidates = []

        for session_index, session in enumerate(self.sessions):
            for detection_index, detection in enumerate(detections):
                score = intersection_over_union(
                    session.bbox, detection
                )

                if score > 0 and score >= self.match_threshold:
                    candidates.append(
                        (score, session_index, detection_index)
                    )

        # Prefer stronger geometric matches.
        candidates.sort(reverse=True)

        matched_sessions = set()
        matched_detections = set()

        for score, session_index, detection_index in candidates:
            if session_index in matched_sessions:
                continue

            if detection_index in matched_detections:
                continue

            session = self.sessions[session_index]
            session.bbox = detections[detection_index]
            session.missed_scans = 0

            matched_sessions.add(session_index)
            matched_detections.add(detection_index)

        # Retain missing sessions for a short grace period.
        retained = []

        for index, session in enumerate(self.sessions):
            if index not in matched_sessions:
                session.missed_scans += 1

            if session.missed_scans < self.max_missed_scans:
                retained.append(session)

        self.sessions = retained

        # Create sessions for previously unseen detections.
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue

            session = GameSession(
                number=self.next_game_number,
                bbox=detection,
            )

            self.next_game_number += 1
            self.sessions.append(session)
