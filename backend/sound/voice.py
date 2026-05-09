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
import paho.mqtt.publish as mqttpublish
from dotenv import load_dotenv
from mistralai.client import Mistral
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
    return convert_to(b"".join(audio_iter), input_sample_rate=22050)

# ---------------------------------------------------------------------------
# MQTT / Core2 interface
# ---------------------------------------------------------------------------

def push_feedback(reps=None, verdict=None, feedback=None, waiting=False, knee_angle=0):
    """push Core2 update, waiting=True pushes waiting screen, then others are not required"""
    from coach import Coach
    c = Coach("/dev/ttyUSB0")  # adjust per OS
    c.push(reps=1, verdict="good", feedback="nice depth", knee_angle=88)
    c.push(reps=2, verdict="shallow", feedback="3 cm above parallel", knee_angle=108)
    
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


def trigger_rep(good: bool = False) -> None:
    """Trigger the rep-completion sound and increment the rep counter on Core2."""
    send_request(TOPIC_TRIGGER_REP, b"good" if good else b"bad")


def display(score: int) -> None:
    """Send a score value to update the Core2 LCD display."""
    # score \in [0,1] -> [red - black - green]
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
