#pragma once

// WiFi
#define WIFI_SSID     ""
#define WIFI_PASSWORD ""

// MQTT
#define MQTT_BROKER   "192.168.1.100"
#define MQTT_PORT     1883
#define MQTT_CLIENT   "m5core2"

// MQTT topics (subscribe)
#define TOPIC_WAV_FILE   "core2/play/file"   // payload: SD filename e.g. "/rep.wav"
#define TOPIC_WAV_DATA   "core2/play/data"   // payload: raw WAV bytes (<=heap available)
#define TOPIC_PCM_START  "core2/play/pcm/start"
#define TOPIC_PCM_DATA   "core2/play/pcm/data"
#define TOPIC_PCM_END    "core2/play/pcm/end"
#define TOPIC_TRIGGER_REP "core2/rep"        // payload: "good" or "bad"
#define TOPIC_SCORE      "core2/score"       // payload: score integer as ASCII

// Speaker
#define SPEAKER_VOLUME  180   // 0–255
#define SPEAKER_CHANNEL 0     // -1 = auto

// SD paths
#define REP_SOUND_PATH       "/rep.wav"
#define REP_GOOD_SOUND_PATH  "/rep_good.wav"
#define INCOMING_WAV    "/incoming.wav"   // temp file written from MQTT WAV data
