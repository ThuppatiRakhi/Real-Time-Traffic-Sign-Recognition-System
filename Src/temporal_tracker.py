from collections import deque


class TemporalTracker:

    def __init__(self, required_frames=3, confidence_threshold=0.80):
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        self.current_candidate_class = None
        self.consecutive_count = 0
        self.consecutive_confidences = []
        self.last_confirmed_class = None

    def update(self, class_id, confidence):

        # Low confidence breaks the consecutive streak
        if confidence < self.confidence_threshold:
            self.consecutive_count = 0
            self.consecutive_confidences = []
            self.current_candidate_class = None
            return {
                "confirmed": False,
                "new_alert": False,
                "class_id": class_id,
                "confidence": confidence,
                "message": "Confidence below threshold."
            }

        # Track consecutive frames of the same class
        if class_id == self.current_candidate_class:
            self.consecutive_count += 1
            self.consecutive_confidences.append(confidence)
        else:
            self.current_candidate_class = class_id
            self.consecutive_count = 1
            self.consecutive_confidences = [confidence]

        # Check if required consecutive frames reached
        if self.consecutive_count >= self.required_frames:
            avg_conf = (
                sum(self.consecutive_confidences[-self.required_frames:])
                / self.required_frames
            )

            # Check if this is a new sign confirmation
            if self.current_candidate_class != self.last_confirmed_class:
                self.last_confirmed_class = self.current_candidate_class
                return {
                    "confirmed": True,
                    "new_alert": True,
                    "class_id": self.current_candidate_class,
                    "confidence": avg_conf,
                    "message": "Sign confirmed."
                }
            else:
                # Same sign still visible and confirmed -> no duplicate alert
                return {
                    "confirmed": True,
                    "new_alert": False,
                    "class_id": self.current_candidate_class,
                    "confidence": avg_conf,
                    "message": "Sign already confirmed."
                }

        return {
            "confirmed": False,
            "new_alert": False,
            "class_id": class_id,
            "confidence": confidence,
            "message": f"Checking sign ({self.consecutive_count}/{self.required_frames} frames)..."
        }

    def reset(self):
        self.current_candidate_class = None
        self.consecutive_count = 0
        self.consecutive_confidences = []
        self.last_confirmed_class = None