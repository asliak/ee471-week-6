import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
import azure.cognitiveservices.speech as speechsdk


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
TTS_DIR = STATIC_DIR / "tts"
ALLOWED_AUDIO_EXTENSIONS = {".wav"}

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TTS_DIR.mkdir(parents=True, exist_ok=True)


def get_speech_config() -> speechsdk.SpeechConfig:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key or not speech_region:
        raise RuntimeError(
            "Missing Azure Speech credentials. Set AZURE_SPEECH_KEY and "
            "AZURE_SPEECH_REGION in your .env file."
        )

    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    config.speech_synthesis_voice_name = "en-US-Ava:DragonHDLatestNeural"
    return config


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"success": False, "error": "Please enter text first."}), 400

    try:
        speech_config = get_speech_config()
        filename = f"{uuid.uuid4().hex}.wav"
        output_path = TTS_DIR / filename
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return jsonify(
                {
                    "success": True,
                    "audio_url": f"/static/tts/{filename}",
                }
            )

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            error_details = getattr(details, "error_details", None) or str(details.reason)
            return jsonify({"success": False, "error": error_details}), 500

        return jsonify({"success": False, "error": "Speech synthesis failed."}), 500
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "Please upload an audio file."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"success": False, "error": "No audio file selected."}), 400

    suffix = Path(audio_file.filename).suffix.lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify(
            {
                "success": False,
                "error": "Please upload a WAV audio file for transcription.",
            }
        ), 400

    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    audio_file.save(saved_path)

    try:
        speech_config = get_speech_config()
        audio_config = speechsdk.audio.AudioConfig(filename=str(saved_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return jsonify({"success": True, "text": result.text})

        if result.reason == speechsdk.ResultReason.NoMatch:
            return jsonify(
                {
                    "success": False,
                    "error": "No speech could be recognized from that file.",
                }
            ), 400

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            error_details = getattr(details, "error_details", None) or str(details.reason)
            return jsonify({"success": False, "error": error_details}), 500

        return jsonify({"success": False, "error": "Speech recognition failed."}), 500
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename: str):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
