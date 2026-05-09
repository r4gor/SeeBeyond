from __future__ import annotations
import os
from voice import trigger_rep, play_sound, transcribe_message, llm_assistant_response, display

_MAX_FEEDBACK  = 256
_INTERIM_VERDICT = "—"


def _validate(reps: int, verdict: str, feedback: str, knee_angle: int | float) -> None:
    if not isinstance(reps, int) or reps < 0:
        raise ValueError(f"reps must be a non-negative int, got {reps!r}")
    if not isinstance(verdict, str) or not verdict.strip():
        raise ValueError(f"verdict must be a non-empty string, got {verdict!r}")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError(f"feedback must be a non-empty string, got {feedback!r}")
    if len(feedback) > _MAX_FEEDBACK:
        raise ValueError(f"feedback too long ({len(feedback)} > {_MAX_FEEDBACK} chars)")
    if not isinstance(knee_angle, (int, float)):
        raise ValueError(f"knee_angle must be numeric, got {knee_angle!r}")
    if not (0 <= knee_angle <= 180):
        raise ValueError(f"knee_angle {knee_angle} out of valid range [0, 180]")


class Coach:
    def __init__(self, broker: str, port: int = 1883):
        """broker: MQTT broker IP/hostname (e.g. '192.168.1.100')."""
        os.environ["MQTT_BROKER"] = broker
        os.environ["MQTT_PORT"]   = str(port)

    def push(self, reps: int, verdict: str, feedback: str, knee_angle: int) -> None:
        _validate(reps, verdict, feedback, knee_angle)

        trigger_rep(good=(verdict == "good"))
        display(reps)

        # Interim messages skip LLM — speak the feedback text directly
        if verdict == _INTERIM_VERDICT:
            wav = transcribe_message(feedback)
        else:
            instruction = llm_assistant_response(
                f"verdict={verdict}, feedback={feedback}, knee_angle={knee_angle}"
            )
            wav = transcribe_message(instruction)

        play_sound(wav)
