from Src.alert_engine import AlertEngine
from Src.temporal_tracker import TemporalTracker


class IntelligentPipeline:

    def __init__(self, confidence_threshold=0.80, required_frames=3):

        self.alert_engine = AlertEngine(
            confidence_threshold=confidence_threshold
        )

        self.temporal_tracker = TemporalTracker(
            required_frames=required_frames
        )

    def process_prediction(self, class_id, confidence):

        # Step 1: Check alert engine
        alert_result = self.alert_engine.process_prediction(
            class_id,
            confidence
        )

        # Low confidence (< 0.80)
        if not alert_result["alert"]:
            self.temporal_tracker.update(class_id, confidence)
            return {
                "confirmed": False,
                "new_alert": False,
                "class_id": class_id,
                "confidence": confidence,
                "name": alert_result["name"],
                "category": alert_result["category"],
                "severity": alert_result["severity"],
                "message": alert_result["message"],
                "voice_message": alert_result.get("voice_message"),
                "is_important": alert_result.get("is_important", False)
            }

        # Step 2: Check temporal consistency
        tracker_result = self.temporal_tracker.update(
            class_id,
            confidence
        )

        is_important = alert_result.get("is_important", False)
        new_alert = tracker_result.get("new_alert", False) and is_important

        return {
            "confirmed": tracker_result["confirmed"],
            "new_alert": new_alert,
            "class_id": class_id,
            "confidence": tracker_result["confidence"],
            "name": alert_result["name"],
            "category": alert_result["category"],
            "severity": alert_result["severity"],
            "message": (
                alert_result["message"]
                if tracker_result["confirmed"]
                else tracker_result["message"]
            ),
            "voice_message": alert_result.get("voice_message"),
            "is_important": is_important
        }

    def reset(self):
        self.temporal_tracker.reset()