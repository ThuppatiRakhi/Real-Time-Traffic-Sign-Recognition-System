from Src.sign_knowledge import SIGN_KNOWLEDGE


class AlertEngine:

    def __init__(self, confidence_threshold=0.80):
        self.confidence_threshold = confidence_threshold

    def process_prediction(self, class_id, confidence):

        # Low-confidence prediction
        if confidence < self.confidence_threshold:
            return {
                "alert": False,
                "status": "UNCERTAIN",
                "class_id": class_id,
                "confidence": confidence,
                "name": "Unknown",
                "category": "Unknown",
                "severity": "Low",
                "message": "Traffic sign detected, but confidence is too low.",
                "voice_message": None,
                "is_important": False
            }

        # Get sign information
        sign = SIGN_KNOWLEDGE.get(class_id)

        if sign is None:
            return {
                "alert": False,
                "status": "UNKNOWN",
                "class_id": class_id,
                "confidence": confidence,
                "name": "Unknown",
                "category": "Unknown",
                "severity": "Low",
                "message": "Unknown traffic sign.",
                "voice_message": None,
                "is_important": False
            }

        return {
            "alert": True,
            "status": "CONFIRMED",
            "class_id": class_id,
            "confidence": confidence,
            "name": sign["name"],
            "category": sign["category"],
            "severity": sign["severity"],
            "message": sign["message"],
            "voice_message": sign.get("voice_message"),
            "is_important": sign.get("is_important", False)
        }