"""
Pipeline:
  classification output (e.g. "shallow, hip 50degree too wide")
    -> llm_assistant_response  : translate to plain instruction
    -> transcribe_message      : ElevenLabs TTS -> WAV bytes
    -> play_sound              : MQTT publish raw WAV to Core2 (core2/play/data)

Core2 MQTT topics:
  core2/play/data  — raw WAV bytes  (triggers immediate playback)
  core2/play/file  — SD filename    (plays a pre-loaded file)
  core2/rep        — empty payload  (triggers /rep.wav + rep counter)
  core2/score      — ASCII integer  (updates score display)
"""

import os
import paho.mqtt.publish as mqttpublish
from dotenv import load_dotenv
from mistralai import Mistral
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MISTRAL_API_KEY     = os.environ["MISTRAL_API_KEY"]
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

MQTT_BROKER = os.environ.get("MQTT_BROKER", "192.168.1.100")
MQTT_PORT   = int(os.environ.get("MQTT_PORT", 1883))

# Topics must match core2/config.h
TOPIC_WAV_DATA    = "core2/play/data"
TOPIC_WAV_FILE    = "core2/play/file"
TOPIC_TRIGGER_REP = "core2/rep"
TOPIC_SCORE       = "core2/score"

_mistral    = Mistral(api_key=MISTRAL_API_KEY)
_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def llm_assistant_response(input: str) -> str:
    """Translate a technical classification string into a plain coaching instruction."""
    response = _mistral.chat.complete(
        model="ministral-8b-latest",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                "You are a fitness instructor. Translate the following technical "
                "classification into a single short coaching instruction a client "
                "would hear during a workout. Reply with only the instruction, "
                "no extra text.\n\n"
                f"Classification: {input}"
            ),
        }],
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------

def transcribe_message(text: str) -> bytes:
    """Generate WAV audio from text using ElevenLabs. Returns raw WAV bytes."""
    audio_iter = _elevenlabs.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.4,
            similarity_boost=0.8,
            style=0.0,
            use_speaker_boost=True,
        ),
        output_format="pcm_22050",  # Core2 speaker works well at 22050 Hz 16-bit PCM
    )
    return b"".join(audio_iter)

# ---------------------------------------------------------------------------
# MQTT / Core2 interface
# ---------------------------------------------------------------------------

def send_request(topic: str, payload: bytes | str, retain: bool = False) -> None:
    """Publish a single MQTT message to the Core2."""
    mqttpublish.single(
        topic,
        payload=payload,
        hostname=MQTT_BROKER,
        port=MQTT_PORT,
        retain=retain,
    )


def play_sound(wav_bytes: bytes) -> None:
    """Send raw WAV bytes to Core2 for immediate playback."""
    send_request(TOPIC_WAV_DATA, wav_bytes)


def play_file(filename: str) -> None:
    """Tell Core2 to play a WAV file already on its SD card (e.g. '/rep.wav')."""
    send_request(TOPIC_WAV_FILE, filename.encode())


def trigger_rep() -> None:
    """Trigger the rep-completion sound and increment the rep counter on Core2."""
    send_request(TOPIC_TRIGGER_REP, b"")


def display(score: int) -> None:
    """Send a score value to update the Core2 LCD display."""
    send_request(TOPIC_SCORE, str(score).encode())


def generate_number_display(number: int) -> None:
    """Alias for display — pushes an integer to the Core2 score screen."""
    display(number)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_classification(classification: str, score: int | None = None) -> None:
    """
    Full pipeline: classification string -> coaching audio on Core2.
    Optionally updates the score display.
    """
    instruction = llm_assistant_response(classification)
    print(f"[voice] instruction: {instruction}")

    wav = transcribe_message(instruction)
    play_sound(wav)

    if score is not None:
        display(score)
