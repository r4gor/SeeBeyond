"""
Run individual stages or the full pipeline.

  python test_sound.py llm
  python test_sound.py tts
  python test_sound.py mqtt
  python test_sound.py pipeline
  python test_sound.py all       # runs every step in order
"""

import sys
import time
import voice

SAMPLE_CLASSIFICATION = "shallow, hip 50degree too wide"

SEP = "-" * 50


def test_llm():
    print(SEP)
    print("TEST: LLM (Mistral)")
    result = voice.llm_assistant_response(SAMPLE_CLASSIFICATION)
    assert isinstance(result, str) and len(result) > 0, "Empty response"
    print(f"  input  : {SAMPLE_CLASSIFICATION}")
    print(f"  output : {result}")
    print("  PASS")
    return result


def test_tts(text: str | None = None):
    print(SEP)
    print("TEST: TTS (ElevenLabs)")
    text = text or "Keep your hips level and squat deeper."
    wav = voice.transcribe_message(text)
    assert len(wav) > 1000, f"WAV too small: {len(wav)} bytes"
    with open("/tmp/test_output.wav", "wb") as f:
        f.write(wav)
    print(f"  text   : {text}")
    print(f"  bytes  : {len(wav)}")
    print(f"  saved  : /tmp/test_output.wav")
    print("  PASS")
    return wav


def test_mqtt(wav: bytes | None = None):
    print(SEP)
    print("TEST: MQTT -> Core2")

    print("  trigger_rep  ...", end=" ", flush=True)
    voice.trigger_rep()
    print("sent")
    time.sleep(0.2)

    print("  display(7)   ...", end=" ", flush=True)
    voice.display(7)
    print("sent")
    time.sleep(0.2)

    print("  play_file    ...", end=" ", flush=True)
    voice.play_file("/rep.wav")
    print("sent")
    time.sleep(0.2)

    if wav:
        print(f"  play_sound ({len(wav)}B) ...", end=" ", flush=True)
        voice.play_sound(wav)
        print("sent")

    print("  PASS (check Core2 screen/speaker)")


def test_pipeline():
    print(SEP)
    print("TEST: Full pipeline")
    voice.process_classification(SAMPLE_CLASSIFICATION, score=5)
    print("  PASS")


TESTS = {
    "llm":      lambda: test_llm(),
    "tts":      lambda: test_tts(),
    "mqtt":     lambda: test_mqtt(),
    "pipeline": lambda: test_pipeline(),
}


def run_all():
    instruction = test_llm()
    wav = test_tts(instruction)
    test_mqtt(wav)
    test_pipeline()
    print(SEP)
    print("ALL PASS")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        run_all()
    elif target in TESTS:
        TESTS[target]()
    else:
        print(f"Unknown target '{target}'. Options: {', '.join(TESTS)} all")
        sys.exit(1)
