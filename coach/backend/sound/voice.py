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
import io
import wave
import hashlib
import threading
from pathlib import Path
import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from mistralai.client import Mistral
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MISTRAL_API_KEY     = os.environ["MISTRAL_API_KEY"]
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

MQTT_BROKER = os.environ.get("MQTT_BROKER", "172.20.10.2")
MQTT_PORT   = int(os.environ.get("MQTT_PORT", 1883))
MQTT_MAX_WAV_PAYLOAD = int(os.environ.get("MQTT_MAX_WAV_PAYLOAD", 60 * 1024))
WAV_HEADER_SIZE = 44
TTS_CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache" / "tts"))

# Topics must match core2/config.h
TOPIC_WAV_DATA    = "core2/play/data"
TOPIC_WAV_FILE    = "core2/play/file"
TOPIC_PCM_START   = "core2/play/pcm/start"
TOPIC_PCM_DATA    = "core2/play/pcm/data"
TOPIC_PCM_END     = "core2/play/pcm/end"
TOPIC_TRIGGER_REP = "core2/rep"
TOPIC_SCORE       = "core2/score"
TOPIC_DISPLAY     = "core2/display"

_mistral    = Mistral(api_key=MISTRAL_API_KEY)
_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ---------------------------------------------------------------------------
# Persistent MQTT client — avoids a TCP handshake on every publish
# ---------------------------------------------------------------------------

_mqtt_lock: threading.Lock = threading.Lock()
_mqtt_client_instance: mqtt.Client | None = None


def _get_client() -> mqtt.Client:
    """Return the shared MQTT client, connecting or reconnecting as needed."""
    global _mqtt_client_instance
    with _mqtt_lock:
        client = _mqtt_client_instance
        if client is not None and client.is_connected():
            return client
        broker, port = _mqtt_settings()
        if client is not None:
            try:
                client.reconnect()
                return client
            except Exception:
                try:
                    client.loop_stop()
                except Exception:
                    pass
                _mqtt_client_instance = None
        new_client = mqtt.Client()
        try:
            new_client.connect(broker, port, keepalive=60)
        except TimeoutError as exc:
            raise ConnectionError(
                f"Timed out connecting to MQTT broker {broker}:{port}. "
                "Check that your broker is running and reachable, or set MQTT_BROKER "
                "and MQTT_PORT to the broker address used by the Core2."
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                f"Could not connect to MQTT broker {broker}:{port}: {exc}. "
                "Check MQTT_BROKER, MQTT_PORT, WiFi, and broker availability."
            ) from exc
        new_client.loop_start()
        _mqtt_client_instance = new_client
        return new_client

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def llm_assistant_response(input: str) -> str:
    """Translate a technical classification string into a plain coaching instruction."""
    response = _mistral.chat.complete(
        model="ministral-3b-latest",
        max_tokens=40,
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
    cached = _read_tts_cache(text)
    if cached is not None:
        return cached

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
    wav_bytes = convert_to(b"".join(audio_iter), input_sample_rate=22050)
    _write_tts_cache(text, wav_bytes)
    return wav_bytes


def _tts_cache_path(text: str) -> Path:
    cache_input = "\n".join([
        "elevenlabs",
        ELEVENLABS_VOICE_ID,
        "eleven_multilingual_v2",
        "pcm_22050",
        "44100hz-16bit-mono-wav",
        "stability=0.4",
        "similarity_boost=0.8",
        "style=0.0",
        "use_speaker_boost=True",
        text,
    ])
    key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
    return TTS_CACHE_DIR / f"{key}.wav"


def _read_tts_cache(text: str) -> bytes | None:
    path = _tts_cache_path(text)
    if not path.exists():
        return None
    return path.read_bytes()


def _write_tts_cache(text: str, wav_bytes: bytes) -> None:
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _tts_cache_path(text)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_bytes(wav_bytes)
    temp_path.replace(path)

# ---------------------------------------------------------------------------
# MQTT / Core2 interface
# ---------------------------------------------------------------------------

def push_feedback(reps=None, verdict=None, feedback=None, waiting=False, knee_angle=0):
    """push Core2 update, waiting=True pushes waiting screen, then others are not required"""
    from coach import Coach
    c = Coach("/dev/ttyUSB0")  # adjust per OS
    c.push(reps=1, verdict="good", feedback="nice depth", knee_angle=88)
    c.push(reps=2, verdict="shallow", feedback="3 cm above parallel", knee_angle=108)


def _mqtt_settings() -> tuple[str, int]:
    broker = os.environ.get("MQTT_BROKER", MQTT_BROKER)
    port = int(os.environ.get("MQTT_PORT", MQTT_PORT))

    if ":" in broker and broker.count(":") == 1:
        broker_host, broker_port = broker.rsplit(":", 1)
        if broker_port.isdigit():
            broker = broker_host
            port = int(broker_port)

    return broker, port


def send_request(topic: str, payload: bytes | str, retain: bool = False) -> None:
    """Publish a single MQTT message to the Core2."""
    _get_client().publish(topic, payload=payload, qos=0, retain=retain)


def play_sound(wav_bytes: bytes) -> None:
    """Send one buffered PCM clip to Core2 for gapless playback."""
    _send_pcm_clip(_extract_pcm_44100_mono_16bit(wav_bytes))


def play_file(filename: str) -> None:
    """Tell Core2 to play a WAV file already on its SD card (e.g. '/rep.wav')."""
    send_request(TOPIC_WAV_FILE, filename.encode())


def trigger_rep(good: bool = False) -> None:
    """Trigger the rep-completion sound on Core2."""
    send_request(TOPIC_TRIGGER_REP, b"good" if good else b"bad")


def send_display_update(
    rep_n: int, verdict: str, feedback: str, angle_deg: int, state: str
) -> None:
    """Push a compact JSON display update to Core2.

    Core2 uses this to update all four display zones in real time.
    JSON keys kept short to fit MQTT comfortably: r=reps, v=verdict,
    f=feedback, a=angle, s=state.
    """
    import json as _json
    payload = _json.dumps({
        "r": rep_n,
        "v": verdict[:14],
        "f": feedback[:45],
        "a": int(angle_deg),
        "s": state[:14],
    })
    send_request(TOPIC_DISPLAY, payload.encode())


def display(score: int) -> None:
    """Send a score value to update the Core2 LCD display."""
    # score \in [0,1] -> background \in [red - black - green]
    send_request(TOPIC_SCORE, str(score).encode())


def generate_number_display(number: int) -> None:
    """Alias for display — pushes an integer to the Core2 score screen."""
    display(number)


def _split_wav_for_mqtt(wav_bytes: bytes) -> list[bytes]:
    if len(wav_bytes) <= MQTT_MAX_WAV_PAYLOAD:
        return [wav_bytes]

    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        params = source.getparams()
        frame_width = params.sampwidth * params.nchannels
        max_pcm_bytes = MQTT_MAX_WAV_PAYLOAD - WAV_HEADER_SIZE
        frames_per_chunk = max(1, max_pcm_bytes // frame_width)

        chunks = []
        while True:
            pcm = source.readframes(frames_per_chunk)
            if not pcm:
                break

            chunk_buffer = io.BytesIO()
            with wave.open(chunk_buffer, "wb") as chunk:
                chunk.setnchannels(params.nchannels)
                chunk.setsampwidth(params.sampwidth)
                chunk.setframerate(params.framerate)
                chunk.writeframes(pcm)
            chunks.append(chunk_buffer.getvalue())

    return chunks


def _send_audio_chunks(chunks: list[bytes]) -> None:
    client = _get_client()
    for chunk in chunks:
        client.publish(TOPIC_WAV_DATA, payload=chunk, qos=0, retain=False)


def _extract_pcm_44100_mono_16bit(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        if source.getframerate() != 44100 or source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError(
                "Core2 audio must be 44100 Hz, 16-bit, mono WAV before PCM transport"
            )
        return source.readframes(source.getnframes())


def _send_pcm_clip(pcm: bytes) -> None:
    client = _get_client()
    max_chunk = MQTT_MAX_WAV_PAYLOAD
    client.publish(TOPIC_PCM_START, str(len(pcm)).encode(), qos=0)
    for offset in range(0, len(pcm), max_chunk):
        client.publish(TOPIC_PCM_DATA, pcm[offset:offset + max_chunk], qos=0)
    client.publish(TOPIC_PCM_END, b"", qos=0).wait_for_publish()

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

# Utils

def _read_wav(audio: bytes | bytearray | memoryview | str | os.PathLike):
    if isinstance(audio, (str, os.PathLike)):
        wav_source = wave.open(str(audio), "rb")
    else:
        wav_source = wave.open(io.BytesIO(bytes(audio)), "rb")

    with wav_source as wav:
        params = wav.getparams()
        return params.framerate, params.sampwidth, params.nchannels, wav.readframes(params.nframes)


def _decode_sample(frame: bytes, sample_width: int) -> int:
    if sample_width == 1:
        return (frame[0] - 128) << 8
    if sample_width == 2:
        return int.from_bytes(frame, "little", signed=True)
    if sample_width == 3:
        sample = int.from_bytes(frame + (b"\xff" if frame[2] & 0x80 else b"\x00"), "little", signed=True)
        return sample >> 8
    if sample_width == 4:
        return int.from_bytes(frame, "little", signed=True) >> 16
    raise ValueError(f"unsupported sample width: {sample_width} bytes")


def _pcm_to_mono_16bit(pcm: bytes, sample_width: int, channels: int) -> list[int]:
    if sample_width < 1:
        raise ValueError("sample_width must be at least 1 byte")
    if channels < 1:
        raise ValueError("channels must be at least 1")

    frame_width = sample_width * channels
    if len(pcm) % frame_width != 0:
        raise ValueError("PCM data length is not aligned to complete frames")

    samples = []
    for offset in range(0, len(pcm), frame_width):
        channel_total = 0
        for channel in range(channels):
            start = offset + channel * sample_width
            channel_total += _decode_sample(pcm[start:start + sample_width], sample_width)
        samples.append(channel_total // channels)
    return samples


def _resample_linear(samples: list[int], source_rate: int, target_rate: int) -> list[int]:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or not samples:
        return samples
    if len(samples) == 1:
        return samples

    target_len = max(1, round(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    resampled = []

    for index in range(target_len):
        source_pos = index * ratio
        left = int(source_pos)
        right = min(left + 1, len(samples) - 1)
        fraction = source_pos - left
        value = samples[left] + (samples[right] - samples[left]) * fraction
        resampled.append(round(value))

    return resampled


def _encode_16bit(samples: list[int]) -> bytes:
    out = bytearray(len(samples) * 2)
    for index, sample in enumerate(samples):
        clamped = max(-32768, min(32767, sample))
        out[index * 2:index * 2 + 2] = clamped.to_bytes(2, "little", signed=True)
    return bytes(out)


def convert_to(
    audio: bytes | bytearray | memoryview | str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    *,
    input_sample_rate: int | None = None,
    input_sample_width: int = 2,
    input_channels: int = 1,
    target_sample_rate: int = 44100,
) -> bytes:
    """Convert audio to a 44100 Hz / 16-bit / mono WAV file.

    `audio` can be WAV bytes, a WAV file path, or raw PCM bytes. For raw PCM,
    pass `input_sample_rate`; `input_sample_width` and `input_channels` default
    to ElevenLabs' `pcm_22050` output shape: 16-bit mono PCM.
    """
    try:
        source_rate, sample_width, channels, pcm = _read_wav(audio)
    except wave.Error:
        if input_sample_rate is None:
            raise ValueError("input_sample_rate is required when converting raw PCM bytes")
        source_rate = input_sample_rate
        sample_width = input_sample_width
        channels = input_channels
        pcm = bytes(audio)

    if sample_width == 2:
        raw = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1).round().astype(np.int16)
        if source_rate != target_sample_rate:
            n_out = max(1, round(len(raw) * target_sample_rate / source_rate))
            raw = np.interp(
                np.linspace(0, len(raw) - 1, n_out),
                np.arange(len(raw)),
                raw.astype(np.float64),
            ).round().clip(-32768, 32767).astype(np.int16)
        converted_pcm = raw.tobytes()
    else:
        samples = _pcm_to_mono_16bit(pcm, sample_width, channels)
        samples = _resample_linear(samples, source_rate, target_sample_rate)
        converted_pcm = _encode_16bit(samples)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(target_sample_rate)
        wav.writeframes(converted_pcm)

    converted = wav_buffer.getvalue()
    if output_path is not None:
        with open(output_path, "wb") as output:
            output.write(converted)
    return converted
