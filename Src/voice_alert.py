import os
import sys
import queue
import threading
import ctypes

_tts_queue = queue.Queue()
_worker_thread = None
_lock = threading.Lock()


def _tts_worker():
    """
    Background worker thread for Text-to-Speech.
    Runs asynchronously and never blocks the main video/webcam loop.
    Properly initializes Windows COM / SAPI audio pipeline.
    """
    try:
        ctypes.windll.ole32.CoInitialize(None)
    except Exception:
        pass

    speaker = None

    # Priority 1: Native Windows SAPI.SpVoice (Direct COM - zero event loop lockups, instant sound)
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = 0      # Normal speech speed
        speaker.Volume = 100  # Full volume
    except Exception as e:
        # Priority 2: pyttsx3 fallback
        try:
            import pyttsx3
            speaker = pyttsx3.init()
            speaker.setProperty("rate", 160)
            speaker.setProperty("volume", 1.0)
        except Exception as e2:
            print(f"[VoiceAlert] All TTS engine initializations failed: {e2}")

    while True:
        try:
            message = _tts_queue.get()
            if message is None:
                _tts_queue.task_done()
                break

            if speaker is not None:
                try:
                    if hasattr(speaker, "Speak"):
                        # Native Windows SAPI: 0 = synchronous within this background thread
                        speaker.Speak(message, 0)
                    elif hasattr(speaker, "say"):
                        speaker.say(message)
                        speaker.runAndWait()
                except Exception as e:
                    print(f"[VoiceAlert] TTS playback error: {e}")
            else:
                print(f"[VoiceAlert] (No audio engine available): \"{message}\"")
        except Exception as e:
            print(f"[VoiceAlert] Worker exception: {e}")
        finally:
            try:
                _tts_queue.task_done()
            except Exception:
                pass

    try:
        ctypes.windll.ole32.CoUninitialize()
    except Exception:
        pass


def _ensure_worker():
    """Ensure the background worker thread is running."""
    global _worker_thread
    with _lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_tts_worker,
                daemon=True,
                name="VoiceAlertWorker"
            )
            _worker_thread.start()


def speak(message: str, clear_backlog: bool = True):
    """
    Speaks a voice message asynchronously.

    Parameters:
    - message: The text string to speak.
    - clear_backlog: If True, clears any pending older messages in queue
                     so only the most recent timely alert is spoken.
    """
    if not message or not isinstance(message, str):
        return

    try:
        _ensure_worker()

        if clear_backlog:
            while not _tts_queue.empty():
                try:
                    _tts_queue.get_nowait()
                    _tts_queue.task_done()
                except (queue.Empty, ValueError):
                    break

        _tts_queue.put(message)
    except Exception as e:
        print(f"[VoiceAlert] Failed to queue alert: {e}")