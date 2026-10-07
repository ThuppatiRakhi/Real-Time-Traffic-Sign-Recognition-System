import os
import sys
import io
import time
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Import local modules
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "Src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sign_detector import detect_sign_candidates
from sign_knowledge import SIGN_KNOWLEDGE
from intelligent_pipeline import IntelligentPipeline
from voice_alert import speak

# streamlit-webrtc & av imports
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


# --------------------------------------------------
# Page Configuration & Global Styling
# --------------------------------------------------
st.set_page_config(
    page_title="Intelligent Traffic Sign Assistant",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Contrast Dark Navy + Cyan ADAS Theme with Strict Font Sizing
GLOBAL_CSS = """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f8fafc;
        font-size: 17px;
    }
    
    /* Hero Header Card */
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #0f1c34 50%, #064e4a 100%);
        border: 2px solid #0284c7;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-main-title {
        font-size: 38px !important;
        font-weight: 900 !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
        color: #38bdf8 !important;
        line-height: 1.25;
    }
    .hero-main-subtitle {
        font-size: 18px !important;
        color: #e2e8f0 !important;
        margin-top: 10px !important;
        font-weight: 500;
        line-height: 1.5;
    }
    
    /* Project Aim Highlight Box */
    .aim-highlight-card {
        background: #0f172a;
        border-left: 6px solid #38bdf8;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 0 14px 14px 0;
        padding: 24px 28px;
        margin: 22px 0;
    }
    .aim-card-title {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 10px;
    }
    .aim-card-text {
        font-size: 18px;
        color: #f1f5f9;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Section & Sub-Headings */
    .section-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
        margin-top: 16px !important;
        margin-bottom: 14px !important;
    }
    .card-title {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 8px !important;
    }
    
    /* Metric Cards */
    .stat-tile {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px 22px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .stat-number {
        font-size: 34px;
        font-weight: 900;
        color: #38bdf8;
        margin: 0;
    }
    .stat-heading {
        font-size: 15px;
        font-weight: 700;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 6px;
    }

    /* Badges */
    .badge-tag {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-critical { background-color: #ef4444; color: #ffffff; }
    .badge-high { background-color: #f97316; color: #ffffff; }
    .badge-medium { background-color: #eab308; color: #0f172a; font-weight: 800; }
    .badge-low { background-color: #22c55e; color: #ffffff; }
    .badge-category { background-color: #0284c7; color: #ffffff; }

    /* Pipeline Flow Diagram */
    .pipeline-wrapper {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        margin: 24px 0;
    }
    .pipeline-node {
        background: #1e293b;
        border: 2px solid #38bdf8;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        font-size: 16px;
        font-weight: 800;
        color: #ffffff;
        flex: 1;
        min-width: 130px;
    }
    .pipeline-arrow-icon {
        font-size: 24px;
        font-weight: 900;
        color: #38bdf8;
    }

    /* Recognition Result Styling */
    .res-sign-title {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
    }
    .res-confidence-val {
        font-size: 34px !important;
        font-weight: 900 !important;
        color: #38bdf8 !important;
    }
    .res-guidance-text {
        font-size: 19px !important;
        color: #f1f5f9 !important;
        line-height: 1.6 !important;
        font-weight: 500;
    }
    .res-voice-text {
        font-size: 18px !important;
        color: #a7f3d0 !important;
        font-weight: 700 !important;
    }

    /* Native Container Card Borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 18px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #38bdf8 !important;
    }
</style>"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# Browser Speech Helper (JS Web Speech API + Python TTS)
# --------------------------------------------------
def trigger_voice_alert(text: str, is_live: bool = False):
    """
    Triggers audible speech on both:
    1. The client browser via JavaScript SpeechSynthesis API
    2. The local host audio engine via speak() (asynchronous daemon)
    """
    if not text:
        return
    
    # 1. Server-side local TTS
    try:
        speak(text)
    except Exception:
        pass

    # 2. Client-side browser SpeechSynthesis
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    js_code = f"""<script>
        (function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{escaped_text}");
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.volume = 1.0;
                msg.lang = 'en-US';
                window.speechSynthesis.speak(msg);
            }}
        }})();
    </script>"""
    components.html(js_code, height=0, width=0)


# --------------------------------------------------
# Model & Asset Loaders
# --------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_roi_traffic_sign_model.keras")

@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_traffic_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    try:
        loaded = load_model(MODEL_PATH)
        return loaded
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_traffic_model()

# Class mapping (GTSRB 43 classes lexicographically sorted as string folders)
CLASS_NAMES = sorted([str(i) for i in range(43)])
INDEX_TO_CLASS = [int(x) for x in CLASS_NAMES]


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def get_severity_badge_html(severity: str) -> str:
    sev_lower = severity.lower() if severity else "low"
    if "crit" in sev_lower:
        return f'<span class="badge-tag badge-critical">{severity}</span>'
    elif "high" in sev_lower:
        return f'<span class="badge-tag badge-high">{severity}</span>'
    elif "med" in sev_lower:
        return f'<span class="badge-tag badge-medium">{severity}</span>'
    else:
        return f'<span class="badge-tag badge-low">{severity}</span>'


def get_meta_sign_image(class_id: int):
    meta_path = os.path.join(BASE_DIR, "Meta", f"{class_id}.png")
    if os.path.exists(meta_path):
        return Image.open(meta_path)
    return None


def run_sign_inference(image_bgr: np.ndarray):
    """
    Executes 2-Stage Pipeline:
    1. HSV + Color Dominance + Shape/Circularity candidate detection
    2. ROI Extraction + 10% Padding + Resize 64x64 + CNN Softmax Inference
    """
    if model is None:
        return {"status": "error", "message": "Model not loaded."}

    candidates = detect_sign_candidates(image_bgr)
    if not candidates:
        return {
            "status": "no_sign",
            "message": "No traffic sign candidate found. Background clutter was rejected."
        }

    # Best candidate bounding box
    candidate = candidates[0]
    x, y, w, h = candidate
    x1, y1, x2, y2 = x, y, x + w, y + h

    # Add 10% padding (consistent with GTSRB training)
    img_h, img_w = image_bgr.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * 0.10)
    pad_y = int(box_h * 0.10)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(img_w, x2 + pad_x)
    crop_y2 = min(img_h, y2 + pad_y)

    sign_crop_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if sign_crop_bgr.size == 0:
        return {"status": "error", "message": "Invalid crop dimensions."}

    sign_crop_rgb = cv2.cvtColor(sign_crop_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for CNN: 64x64, RGB, float32 / 255.0
    resized = cv2.resize(sign_crop_rgb, (64, 64))
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Predict
    raw_predictions = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(raw_predictions))
    confidence = float(raw_predictions[predicted_idx])
    class_id = INDEX_TO_CLASS[predicted_idx]

    # Top-5 breakdown
    top5_indices = np.argsort(raw_predictions)[::-1][:5]
    top5_results = []
    for idx in top5_indices:
        cid = INDEX_TO_CLASS[idx]
        c_conf = float(raw_predictions[idx])
        c_name = SIGN_KNOWLEDGE.get(cid, {}).get("name", f"Class {cid}")
        top5_results.append({"class_id": cid, "name": c_name, "confidence": c_conf})

    # Annotate original image with bounding box & label
    annotated_bgr = image_bgr.copy()
    sign_info = SIGN_KNOWLEDGE.get(class_id, {
        "name": f"Class {class_id}",
        "category": "General",
        "severity": "Low",
        "message": "Traffic sign detected.",
        "voice_message": None,
        "is_important": False
    })
    label_text = f"{sign_info['name']} ({confidence*100:.1f}%)"
    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(annotated_bgr, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 200, 0), 1)
    
    # Label banner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(annotated_bgr, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(annotated_bgr, label_text, (x1 + 5, max(th + 5, y1 - 5)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "status": "success",
        "class_id": class_id,
        "confidence": confidence,
        "sign_info": sign_info,
        "annotated_image": annotated_rgb,
        "cropped_roi": sign_crop_rgb,
        "top5": top5_results,
        "bbox": (x1, y1, x2, y2)
    }


# --------------------------------------------------
# WebRTC Configuration & Video Processor
# --------------------------------------------------
if HAS_WEBRTC:
    def get_rtc_configuration() -> RTCConfiguration:
        """
        Constructs an RTCConfiguration with Google STUN servers as fallback
        and Metered TURN servers parsed from environment variables.
        """
        ice_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]

        turn_urls_env = os.environ.get("TURN_URLS", "").strip()
        turn_username = os.environ.get("TURN_USERNAME", "").strip()
        turn_credential = os.environ.get("TURN_CREDENTIAL", "").strip()

        default_metered_turn_urls = [
            "turn:global.relay.metered.ca:80",
            "turn:global.relay.metered.ca:80?transport=tcp",
            "turn:global.relay.metered.ca:443",
            "turns:global.relay.metered.ca:443?transport=tcp",
        ]

        if turn_username and turn_credential:
            if turn_urls_env:
                if turn_urls_env.startswith("[") and turn_urls_env.endswith("]"):
                    try:
                        parsed = json.loads(turn_urls_env)
                        if isinstance(parsed, list):
                            turn_urls = [str(u).strip() for u in parsed if str(u).strip()]
                        else:
                            turn_urls = [u.strip() for u in turn_urls_env.split(",") if u.strip()]
                    except Exception:
                        turn_urls = [u.strip() for u in turn_urls_env.split(",") if u.strip()]
                else:
                    turn_urls = [u.strip() for u in turn_urls_env.split(",") if u.strip()]
            else:
                turn_urls = default_metered_turn_urls

            if turn_urls:
                ice_servers.append(
                    {
                        "urls": turn_urls,
                        "username": turn_username,
                        "credential": turn_credential,
                    }
                )

        return RTCConfiguration({"iceServers": ice_servers})

    RTC_CONFIGURATION = get_rtc_configuration()

    class TrafficSignVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.pipeline = IntelligentPipeline(confidence_threshold=0.80, required_frames=3)
            self.no_sign_count = 0
            self.voice_on = True

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            img_h, img_w = img_bgr.shape[:2]

            candidates = detect_sign_candidates(img_bgr)

            if not candidates:
                self.no_sign_count += 1
                if self.no_sign_count >= 5:
                    self.pipeline.reset()

                annotated = img_bgr.copy()
                cv2.putText(
                    annotated,
                    "Searching for traffic signs...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                self.no_sign_count = 0
                cand = candidates[0]
                x, y, w, h = cand
                x1, y1, x2, y2 = x, y, x + w, y + h

                # 10% padding (consistent with GTSRB training & image inference)
                px = int(w * 0.10)
                py = int(h * 0.10)
                cx1 = max(0, x1 - px)
                cy1 = max(0, y1 - py)
                cx2 = min(img_w, x2 + px)
                cy2 = min(img_h, y2 + py)

                crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
                if crop_bgr.size > 0 and model is not None:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (64, 64))
                    crop_norm = crop_resized.astype(np.float32) / 255.0
                    input_t = np.expand_dims(crop_norm, axis=0)

                    preds = model.predict(input_t, verbose=0)[0]
                    pred_idx = int(np.argmax(preds))
                    conf = float(preds[pred_idx])
                    cid = INDEX_TO_CLASS[pred_idx]

                    res = self.pipeline.process_prediction(cid, conf)

                    # Speak on confirmation (background TTS thread, no Streamlit UI calls)
                    if res["new_alert"] and res["is_important"] and self.voice_on:
                        v_text = res.get("voice_message") or res.get("message")
                        if v_text:
                            try:
                                speak(v_text)
                            except Exception:
                                pass

                    # Visual overlay
                    annotated = img_bgr.copy()
                    box_col = (0, 255, 0) if res["confirmed"] else (0, 220, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, 3)

                    lbl = f"{res['name']} ({conf*100:.1f}%)" if res["confirmed"] else f"Checking ({conf*100:.1f}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(lbl, font, 0.65, 2)
                    cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_col, -1)
                    cv2.putText(annotated, lbl, (x1 + 4, max(th + 4, y1 - 4)), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    annotated = img_bgr.copy()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
else:
    RTC_CONFIGURATION = None
    TrafficSignVideoProcessor = None


# --------------------------------------------------
# Sidebar (Compact & Professional)
# --------------------------------------------------
with st.sidebar:
    st.markdown("""<div style="text-align: center; padding: 10px 0 16px 0;">
        <h2 style="margin: 0; font-size: 24px; font-weight: 800; color: #38bdf8;">🚦 TrafficAI Assistant</h2>
        <p style="margin: 4px 0 0 0; font-size: 13px; color: #cbd5e1; font-weight: 500;">ADAS Safety & Guidance</p>
    </div>""", unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📹 Live Camera",
            "📷 Image Recognition",
            "🚦 Traffic Sign Library",
            "🧠 AI Model",
            "ℹ️ About"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # System Status Card
    st.markdown("""<div style="background: #0f172a; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
        <p style="margin: 0 0 10px 0; font-size: 12px; font-weight: 800; color: #94a3b8; text-transform: uppercase;">System Engine Status</p>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="height: 10px; width: 10px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
            <span style="font-size: 14px; color: #ffffff;">CNN Model: <b>Active</b></span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="height: 10px; width: 10px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
            <span style="font-size: 14px; color: #ffffff;">Sign Detector: <b>Ready</b></span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="height: 10px; width: 10px; background-color: #38bdf8; border-radius: 50%; display: inline-block;"></span>
            <span style="font-size: 14px; color: #ffffff;">Classes: <b>43 GTSRB</b></span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="height: 10px; width: 10px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
            <span style="font-size: 14px; color: #ffffff;">Voice Engine: <b>Ready</b></span>
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<br><div style='font-size: 12px; color: #64748b; text-align: center;'>ADAS Safety AI Platform</div>", unsafe_allow_html=True)


# ==================================================
# 1. 🏠 DASHBOARD
# ==================================================
if app_mode == "🏠 Dashboard":
    st.markdown("""<div class="hero-banner">
        <h1 class="hero-main-title">🚦 INTELLIGENT TRAFFIC SIGN ASSISTANT</h1>
        <p class="hero-main-subtitle">
            AI-powered traffic sign detection, recognition and voice assistance for safer driving.
        </p>
    </div>""", unsafe_allow_html=True)

    # Project Aim Card
    st.markdown("""<div class="aim-highlight-card">
        <div class="aim-card-title">🎯 PROJECT AIM</div>
        <div class="aim-card-text">
            "To develop an intelligent computer-vision-based driver assistance system that detects traffic signs in real time, 
            classifies them using deep learning, and provides timely safety guidance through voice alerts."
        </div>
    </div>""", unsafe_allow_html=True)

    # How The System Works
    st.markdown('<div class="section-title">🔄 HOW THE SYSTEM WORKS</div>', unsafe_allow_html=True)
    st.markdown("""<div class="pipeline-wrapper">
        <div class="pipeline-node">1. 📹 Live Camera</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node">2. 👁️ Sign Detection</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node">3. ✂️ ROI Extraction</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node">4. 🧠 CNN Classification</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node">5. ✓ 3-Frame Confirmation</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node" style="border-color: #22c55e; color: #4ade80;">6. 🔊 Voice Alert</div>
        <div class="pipeline-arrow-icon">➔</div>
        <div class="pipeline-node" style="border-color: #22c55e; color: #4ade80;">7. 🛡️ Safety Guidance</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Project Statistics (Exact Verified Numbers)
    st.markdown('<div class="section-title">📊 PROJECT BENCHMARK STATISTICS</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        with st.container(border=True):
            st.markdown("<div class='stat-number'>43</div><div class='stat-heading'>Sign Classes</div>", unsafe_allow_html=True)
    with m2:
        with st.container(border=True):
            st.markdown("<div class='stat-number'>39,209</div><div class='stat-heading'>Training Images</div>", unsafe_allow_html=True)
    with m3:
        with st.container(border=True):
            st.markdown("<div class='stat-number'>7,842</div><div class='stat-heading'>Validation Images</div>", unsafe_allow_html=True)
    with m4:
        with st.container(border=True):
            st.markdown("<div class='stat-number'>6,549</div><div class='stat-heading'>Test Evaluated</div>", unsafe_allow_html=True)
    with m5:
        with st.container(border=True):
            st.markdown("<div class='stat-number' style='color:#22c55e;'>99.97%</div><div class='stat-heading'>Best Val Accuracy</div>", unsafe_allow_html=True)
    with m6:
        with st.container(border=True):
            st.markdown("<div class='stat-number' style='color:#22c55e;'>98.66%</div><div class='stat-heading'>Test Accuracy</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Quick Jump Cards
    st.markdown('<div class="section-title">🚀 QUICK ACCESS</div>', unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        with st.container(border=True):
            st.markdown('<div class="card-title">📹 Live Camera Feed</div>', unsafe_allow_html=True)
            st.markdown("Start the continuous live webcam stream with 3-frame confirmation and audible voice alerts.")
    with qc2:
        with st.container(border=True):
            st.markdown('<div class="card-title">📷 Image Recognition</div>', unsafe_allow_html=True)
            st.markdown("Upload road scene photos or test 1-click presets with instant candidate detection, classification & voice alert.")
    with qc3:
        with st.container(border=True):
            st.markdown('<div class="card-title">🚦 Traffic Sign Library</div>', unsafe_allow_html=True)
            st.markdown("Explore all 43 GTSRB benchmark traffic sign categories, safety descriptions, and voice scripts.")


# ==================================================
# 2. 📹 LIVE CAMERA
# ==================================================
elif app_mode == "📹 Live Camera":
    st.markdown("""<div class="hero-banner" style="padding: 26px 32px;">
        <h1 class="hero-main-title">📹 LIVE TRAFFIC SIGN DETECTION</h1>
        <p class="hero-main-subtitle">
            Real-time traffic sign recognition with intelligent voice assistance.
        </p>
    </div>""", unsafe_allow_html=True)

    if not HAS_WEBRTC:
        st.error("⚠️ `streamlit-webrtc` or `av` is not installed in the environment.")
    else:
        # Controls Bar
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1:
            st.markdown("Click **START** below to allow browser webcam access and stream in real time.")
        with c_col2:
            voice_on = st.checkbox("🔊 Voice Alerts Enabled", value=True)

        st.markdown("---")

        v_col, info_col = st.columns([1.6, 1.2])

        with v_col:
            ctx = webrtc_streamer(
                key="traffic-sign-live-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=TrafficSignVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.voice_on = voice_on

        with info_col:
            with st.container(border=True):
                st.markdown('<div class="card-title" style="color: #22c55e;">🟢 SYSTEM READY</div>', unsafe_allow_html=True)
                st.markdown("**Detection Status:** Active Stream")
                st.markdown("""
                Point your webcam towards a traffic sign (e.g., STOP, Speed Limit, Yield, No Entry).
                
                - **Candidate Filtering:** Stage-1 OpenCV detector isolates real sign candidates and filters background clutter.
                - **3-Frame Confirmation:** Guarantees temporal consistency before triggering voice alerts.
                - **Non-Blocking Audio:** Audible voice alert is triggered once upon sign confirmation.
                - **Visual Overlay:** Bounding boxes, predicted sign class, and confidence are rendered directly on the stream.
                """)


# ==================================================
# 3. 📷 IMAGE RECOGNITION
# ==================================================
elif app_mode == "📷 Image Recognition":
    st.markdown("""<div class="hero-banner" style="padding: 26px 32px;">
        <h1 class="hero-main-title">📷 TRAFFIC SIGN RECOGNITION</h1>
        <p class="hero-main-subtitle">
            Upload road scene images or select benchmark presets for instant AI analysis and voice alert.
        </p>
    </div>""", unsafe_allow_html=True)

    input_choice = st.radio("Select Image Source:", ["📁 Upload Image File", "✨ Quick Benchmark Presets"], horizontal=True)
    img_bgr = None
    input_source_key = ""

    if input_choice == "📁 Upload Image File":
        uploaded = st.file_uploader("Upload road scene image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_i = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_i), cv2.COLOR_RGB2BGR)
            input_source_key = f"uploaded_{uploaded.name}_{uploaded.size}"

    elif input_choice == "✨ Quick Benchmark Presets":
        st.markdown('<div class="section-title" style="font-size: 22px !important;">Select a Benchmark Sign to Test:</div>', unsafe_allow_html=True)
        p_cols = st.columns(6)
        presets = [
            (14, "STOP Sign"),
            (13, "Yield Sign"),
            (5, "Speed 80 km/h"),
            (1, "Speed 30 km/h"),
            (17, "No Entry"),
            (12, "Priority Road")
        ]
        for i, (cid, sname) in enumerate(presets):
            with p_cols[i]:
                with st.container(border=True):
                    m_icon = get_meta_sign_image(cid)
                    if m_icon:
                        st.image(m_icon, use_container_width=True)
                    if st.button(f"{sname}", key=f"btn_preset_{cid}", use_container_width=True):
                        st.session_state["chosen_preset"] = cid

        p_id = st.session_state.get("chosen_preset", None)
        if p_id is not None:
            m_path = os.path.join(BASE_DIR, "Meta", f"{p_id}.png")
            if os.path.exists(m_path):
                pil_img = Image.open(m_path).convert("RGBA")
                bg = Image.new("RGBA", pil_img.size, (255, 255, 255))
                composed = Image.alpha_composite(bg, pil_img).convert("RGB")
                raw = cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)
                canvas = np.ones((320, 480, 3), dtype=np.uint8) * 210
                cv2.rectangle(canvas, (0, 220), (480, 320), (90, 90, 95), -1)
                canvas[40:180, 170:310] = cv2.resize(raw, (140, 140))
                img_bgr = canvas
                input_source_key = f"preset_{p_id}"

    if img_bgr is not None:
        with st.spinner("Executing 2-Stage Recognition Pipeline..."):
            res = run_sign_inference(img_bgr)

        if res["status"] == "no_sign":
            st.warning("⚠️ **No Traffic Sign Detected:** The Stage-1 candidate filter analyzed the image and confirmed no traffic signs are present. Background clutter was rejected.")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Input Scene (No Valid Signs Found)", use_container_width=True)

        elif res["status"] == "success":
            sign_info = res["sign_info"]
            conf = res["confidence"]
            class_id = res["class_id"]

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 RECOGNITION RESULTS</div>', unsafe_allow_html=True)

            # Tri-View Inspection
            c1, c2, c3 = st.columns([1.4, 1, 1])
            with c1:
                with st.container(border=True):
                    st.markdown('<div class="card-title">1. Analyzed Scene</div>', unsafe_allow_html=True)
                    st.image(res["annotated_image"], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown('<div class="card-title">2. Cropped Sign ROI</div>', unsafe_allow_html=True)
                    st.image(res["cropped_roi"], use_container_width=True)
            with c3:
                with st.container(border=True):
                    st.markdown('<div class="card-title">3. GTSRB Reference</div>', unsafe_allow_html=True)
                    ref_icon = get_meta_sign_image(class_id)
                    if ref_icon:
                        st.image(ref_icon, use_container_width=True)
                    else:
                        st.write(f"Class {class_id}")

            # Dominant Result Box
            with st.container(border=True):
                r_top1, r_top2 = st.columns([2, 1])
                with r_top1:
                    st.markdown(f"<div class='res-sign-title'>{sign_info['name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"""<div style="margin-top: 10px;">
                        <span class="badge-tag badge-category">{sign_info['category']}</span>
                        {get_severity_badge_html(sign_info['severity'])}
                        <span style="color: #cbd5e1; font-size: 16px; margin-left: 10px; font-weight: 700;">GTSRB CLASS #{class_id}</span>
                    </div>""", unsafe_allow_html=True)
                with r_top2:
                    st.markdown(f"<div class='res-confidence-val' style='text-align: right;'>{conf*100:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<div style='text-align: right; color: #cbd5e1; font-size: 15px; font-weight: 700;'>CONFIDENCE SCORE</div>", unsafe_allow_html=True)

                st.progress(min(1.0, conf))

                st.markdown("---")
                st.markdown('<div style="font-size: 22px; font-weight: 700; color: #38bdf8;">🛡️ DRIVER SAFETY GUIDANCE</div>', unsafe_allow_html=True)
                st.markdown(f"<div class='res-guidance-text'>{sign_info['message']}</div>", unsafe_allow_html=True)

                voice_msg = sign_info.get("voice_message") or sign_info.get("message")
                is_voice_appropriate = (conf >= 0.80) and sign_info.get("is_important", False)

                if is_voice_appropriate and voice_msg:
                    st.markdown("---")
                    st.markdown('<div style="font-size: 22px; font-weight: 700; color: #4ade80;">🔊 VOICE ALERT</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='res-voice-text'>\"{voice_msg}\"</div>", unsafe_allow_html=True)
                    
                    # Voice Trigger with Debounce to prevent continuous repeat on rerun
                    current_event_key = f"{input_source_key}_{class_id}"
                    last_spoken = st.session_state.get("last_spoken_image_event", "")
                    
                    if current_event_key != last_spoken:
                        trigger_voice_alert(voice_msg, is_live=False)
                        st.session_state["last_spoken_image_event"] = current_event_key
                    
                    # Replay Voice Button
                    if st.button("🔊 Replay Voice Alert", key="btn_replay_voice"):
                        trigger_voice_alert(voice_msg, is_live=False)

            # Top-5 Probability Breakdown
            st.markdown("---")
            st.markdown('<div class="section-title">📊 MODEL PROBABILITY DISTRIBUTION — TOP 5</div>', unsafe_allow_html=True)
            st.info("The CNN evaluates all 43 GTSRB classes. These are the five classes with the highest predicted probabilities. Only the highest-confidence class is the final prediction.")

            for item in res["top5"]:
                p_val = item["confidence"] * 100
                st.markdown(f"<div style='font-size: 17px;'><b>{item['name']} (Class #{item['class_id']})</b> — <code>{p_val:.2f}%</code></div>", unsafe_allow_html=True)
                st.progress(min(1.0, item["confidence"]))


# ==================================================
# 4. 🚦 TRAFFIC SIGN LIBRARY
# ==================================================
elif app_mode == "🚦 Traffic Sign Library":
    st.markdown("""<div class="hero-banner" style="padding: 26px 32px;">
        <h1 class="hero-main-title">🚦 TRAFFIC SIGN LIBRARY</h1>
        <p class="hero-main-subtitle">
            Comprehensive catalog of all 43 GTSRB benchmark traffic signs with safety metadata.
        </p>
    </div>""", unsafe_allow_html=True)

    # Filter Controls
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search_kw = st.text_input("🔍 Search signs by name or Class ID...", "").strip().lower()
    with fc2:
        all_cats = ["All Categories"] + sorted(list(set(info["category"] for info in SIGN_KNOWLEDGE.values())))
        chosen_cat = st.selectbox("Filter Category", all_cats)
    with fc3:
        all_sevs = ["All Severities", "Critical", "High", "Medium", "Low"]
        chosen_sev = st.selectbox("Filter Severity", all_sevs)

    # Filter data
    matched = []
    for cid in range(43):
        inf = SIGN_KNOWLEDGE.get(cid, {})
        n = inf.get("name", f"Class {cid}")
        c = inf.get("category", "General")
        s = inf.get("severity", "Low")

        if search_kw and (search_kw not in n.lower() and search_kw != str(cid)):
            continue
        if chosen_cat != "All Categories" and c != chosen_cat:
            continue
        if chosen_sev != "All Severities" and s.lower() != chosen_sev.lower():
            continue

        matched.append((cid, inf))

    st.markdown(f"**Displaying {len(matched)} of 43 Traffic Signs:**")

    # 4 Cards Per Row Grid
    row_size = 4
    for i in range(0, len(matched), row_size):
        row_signs = matched[i : i + row_size]
        cols = st.columns(row_size)
        for c_idx, (cid, inf) in enumerate(row_signs):
            with cols[c_idx]:
                with st.container(border=True):
                    # Header with ID and Severity
                    st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 15px; font-weight: 800; color: #cbd5e1;">CLASS #{cid}</span>
                        {get_severity_badge_html(inf.get('severity', 'Low'))}
                    </div>""", unsafe_allow_html=True)

                    # Sign Image (Inside Card Container)
                    meta_pic = get_meta_sign_image(cid)
                    if meta_pic:
                        st.image(meta_pic, width=110)
                    else:
                        st.write("🚦")

                    # Title & Description
                    st.markdown(f'<div class="card-title">{inf.get("name")}</div>', unsafe_allow_html=True)
                    st.markdown(f"<span class='badge-tag badge-category'>{inf.get('category')}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Safety Guidance:** {inf.get('message')}")
                    
                    if inf.get("is_important"):
                        st.markdown("🔊 **Voice Alert:** Active")
                    else:
                        st.caption("ℹ️ Visual Alert Only")


# ==================================================
# 5. 🧠 AI MODEL
# ==================================================
elif app_mode == "🧠 AI Model":
    st.markdown("""<div class="hero-banner" style="padding: 26px 32px;">
        <h1 class="hero-main-title">🧠 AI MODEL & ARCHITECTURE</h1>
        <p class="hero-main-subtitle">
            Detailed inspection of the trained Convolutional Neural Network and GTSRB benchmark evaluation.
        </p>
    </div>""", unsafe_allow_html=True)

    col_arch, col_stats = st.columns([1.3, 1])

    with col_arch:
        st.markdown('<div class="section-title">🏗️ Loaded Keras Model Summary</div>', unsafe_allow_html=True)
        if model is not None:
            sum_lines = []
            model.summary(print_fn=sum_lines.append)
            raw_summary = "\\n".join(sum_lines)
            st.code(raw_summary, language="text")
        else:
            st.error("Model file could not be loaded.")

    with col_stats:
        st.markdown('<div class="section-title">📈 Verified Evaluation Metrics</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            | Parameter | Verified Value |
            | :--- | :--- |
            | **Benchmark Dataset** | GTSRB (German Traffic Signs) |
            | **Number of Classes** | 43 Classes |
            | **Training Images** | 39,209 ROIs |
            | **Validation Images** | 7,842 ROIs |
            | **Test Images Evaluated** | 6,549 ROIs |
            | **Input Tensor Shape** | 64 × 64 × 3 RGB |
            | **Input Normalization** | [0, 1] Range (/255.0) |
            | **Model File** | `models/best_roi_traffic_sign_model.keras` |
            | **Best Validation Accuracy** | **99.97%** |
            | **Test Accuracy** | **98.66%** |
            """)


# ==================================================
# 6. ℹ️ ABOUT
# ==================================================
elif app_mode == "ℹ️ About":
    st.markdown("""<div class="hero-banner" style="padding: 26px 32px;">
        <h1 class="hero-main-title">ℹ️ ABOUT THE SYSTEM</h1>
        <p class="hero-main-subtitle">
            Advanced Driver Assistance System (ADAS) engineered for autonomous vehicles and driver safety.
        </p>
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title" style="margin-top: 0 !important;">🎯 Project Aim</div>', unsafe_allow_html=True)
        st.markdown("""
        "To develop an intelligent computer-vision-based driver assistance system that detects traffic signs in real time, 
        classifies them using deep learning, and provides timely safety guidance through voice alerts."
        """)

    st.markdown("---")
    st.markdown('<div class="section-title">⚠️ Problem Being Addressed & Solution</div>', unsafe_allow_html=True)
    prob_col, sol_col = st.columns(2)
    with prob_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">⚠️ The Challenge</div>', unsafe_allow_html=True)
            st.markdown("""
            Driver inattention, speed limit changes, and obscured signage cause significant road hazards. 
            Standard pure-CNN detectors suffer from severe false positives when exposed to random background clutter because they lack candidate validation.
            """)
    with sol_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">💡 The Solution</div>', unsafe_allow_html=True)
            st.markdown("""
            A 2-stage vision architecture: Stage-1 isolates valid sign contours via color dominance and circularity/solidity filters; 
            Stage-2 classifies 64×64 ROIs with a deep CNN, backed by 3-frame temporal confirmation and audible voice guidance.
            """)

    st.markdown("---")
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    techs = [
        ("Python", "3.11 Runtime", "🐍"),
        ("TensorFlow / Keras", "2.18 Deep Learning", "🧠"),
        ("OpenCV", "4.11 Computer Vision", "👁️"),
        ("Streamlit", "Web Interface", "⚡"),
        ("NumPy", "Tensor Operations", "🔢"),
        ("Pyttsx3 / SAPI", "Audio Voice Engine", "🔊")
    ]
    for i, (tn, tv, ti) in enumerate(techs):
        with [t1, t2, t3, t4, t5, t6][i]:
            with st.container(border=True):
                st.markdown(f"<div style='text-align:center;'><div style='font-size:28px;'>{ti}</div><div style='font-weight:700; color:#38bdf8;'>{tn}</div><div style='font-size:13px; color:#94a3b8;'>{tv}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🌟 Key Engineering Features & Limitations</div>', unsafe_allow_html=True)
    f_col, l_col = st.columns(2)
    with f_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">🌟 Key Innovations</div>', unsafe_allow_html=True)
            st.markdown("""
            1. **2-Stage Candidate Detection:** HSV segmentation and geometric circularity/solidity tests reject background noise before inference.
            2. **3-Frame Temporal Confirmation:** Filters out transient camera shake and motion blur.
            3. **Dual Voice Alerts:** JavaScript Web Speech API for browser playback + Python SAPI for local hardware audio.
            4. **43 GTSRB Classes:** Complete European benchmark coverage.
            """)
    with l_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">⚠️ Operational Limitations</div>', unsafe_allow_html=True)
            st.markdown("""
            1. **Extreme Low-Light Conditions:** Highly degraded lighting can reduce contour segmentation accuracy.
            2. **Partial Occlusions > 60%:** Severe physical obstructions may prevent geometric candidate extraction.
            3. **Non-GTSRB Signs:** Non-standard regional road signs are mapped to closest geometric class or filtered as background.
            """)
