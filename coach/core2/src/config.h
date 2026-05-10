#pragma once

// WiFi
#define WIFI_SSID     "iPhone.exe"
#define WIFI_PASSWORD "studiostudio"

// MQTT
#define MQTT_BROKER   "172.20.10.2"
#define MQTT_PORT     1883
#define MQTT_CLIENT   "m5core2"

// ElevenLabs TTS — copy ELEVENLABS_API_KEY from coach/backend/.env
#define ELEVENLABS_API_KEY   "sk_6fd3190b6735c723ff09660d17581a639273b8ddc49db899"
#define ELEVENLABS_VOICE_ID  "EXAVITQu4vr4xnSDxMaL"
#define ELEVENLABS_MODEL_ID  "eleven_flash_v2"

// MQTT topics (subscribe)
#define TOPIC_TTS_SPEAK  "core2/tts/speak"   // payload: UTF-8 text → Core2 calls ElevenLabs
#define TOPIC_WAV_FILE   "core2/play/file"   // payload: SD filename e.g. "/rep.wav"
#define TOPIC_WAV_DATA   "core2/play/data"   // payload: raw WAV bytes (<=heap available)
#define TOPIC_PCM_START  "core2/play/pcm/start"
#define TOPIC_PCM_DATA   "core2/play/pcm/data"
#define TOPIC_PCM_END    "core2/play/pcm/end"
#define TOPIC_TRIGGER_REP "core2/rep"        // payload: "good" or "bad"
#define TOPIC_SCORE      "core2/score"       // payload: score integer as ASCII
#define TOPIC_DISPLAY    "core2/display"     // payload: JSON {r,v,f,a,s}

// Speaker
#define SPEAKER_VOLUME  180   // 0–255
#define SPEAKER_CHANNEL 0     // -1 = auto

// SD paths
#define REP_SOUND_PATH       "/incorrect.wav"
#define REP_GOOD_SOUND_PATH  "/correct.wav"
#define INCOMING_WAV    "/incoming.wav"   // temp file written from MQTT WAV data
