#include "mqtt_handler.h"
#include "wav_player.h"
#include "tts_client.h"
#include "config.h"
#include <WiFi.h>
#include <SD.h>

static PubSubClient* _client = nullptr;

extern void drawStatus(const char* msg);
extern void onScoreReceived(int n);

static const size_t MAX_WAV_PAYLOAD = 60 * 1024;

// ---------------------------------------------------------------------------

static void onMessage(const char* topic, byte* payload, unsigned int length) {
    // core2/play/pcm/start — new audio cue incoming; flush stale chunks and pre-buffer
    if (strcmp(topic, TOPIC_PCM_START) == 0) {
        Serial.println("[PCM] stream start");
        drawStatus("PCM streaming");
        startAudioStream();
        return;
    }

    // core2/play/pcm/data — raw PCM chunk; enqueue immediately for playback
    if (strcmp(topic, TOPIC_PCM_DATA) == 0) {
        uint8_t* copy = (uint8_t*)malloc(length);
        if (!copy) {
            Serial.printf("[PCM] malloc failed for chunk %u bytes\n", length);
            return;
        }
        memcpy(copy, payload, length);
        queueAudioChunk(copy, length);
        return;
    }

    // core2/play/pcm/end — stream complete; release pre-buffer so short cues play
    if (strcmp(topic, TOPIC_PCM_END) == 0) {
        Serial.println("[PCM] stream end");
        endAudioStream();
        return;
    }

    // core2/play/file — SD filename; routed through audio task to avoid conflicts
    if (strcmp(topic, TOPIC_WAV_FILE) == 0) {
        char filename[64] = {0};
        size_t n = length < sizeof(filename) - 1 ? length : sizeof(filename) - 1;
        memcpy(filename, payload, n);
        Serial.printf("[MQTT] play file: %s\n", filename);
        drawStatus("MQTT play file");
        queueWAVFromSD(filename);
        return;
    }

    // core2/play/data — raw WAV bytes; routed through audio task
    if (strcmp(topic, TOPIC_WAV_DATA) == 0) {
        if (length > MAX_WAV_PAYLOAD) {
            Serial.printf("[MQTT] WAV payload too large: %u\n", length);
            drawStatus("WAV too large");
            return;
        }
        Serial.printf("[MQTT] WAV data received: %u bytes\n", length);
        drawStatus("MQTT WAV received");
        queueWAVBuffer((const uint8_t*)payload, length);
        return;
    }

    // core2/score — set rep counter to the given integer
    if (strcmp(topic, TOPIC_SCORE) == 0) {
        char buf[16] = {0};
        size_t n = length < sizeof(buf) - 1 ? length : sizeof(buf) - 1;
        memcpy(buf, payload, n);
        int score = atoi(buf);
        Serial.printf("[MQTT] score: %d\n", score);
        onScoreReceived(score);
        return;
    }

    // core2/tts/speak — text payload; Core2 calls ElevenLabs and plays result
    if (strcmp(topic, TOPIC_TTS_SPEAK) == 0) {
        char text[256] = {0};
        size_t n = length < sizeof(text) - 1 ? length : sizeof(text) - 1;
        memcpy(text, payload, n);
        Serial.printf("[MQTT] tts: %s\n", text);
        scheduleTTS(text);
        return;
    }

    // core2/rep — rep sound routed through audio task to avoid blocking MQTT
    if (strcmp(topic, TOPIC_TRIGGER_REP) == 0) {
        bool good = (length == 4 && strncmp((char*)payload, "good", 4) == 0);
        Serial.printf("[MQTT] trigger rep good=%d\n", good);
        drawStatus(good ? "MQTT good rep" : "MQTT bad rep");
        queueWAVFromSD(good ? REP_GOOD_SOUND_PATH : REP_SOUND_PATH);
        onRepReceived();
        return;
    }
}

// ---------------------------------------------------------------------------

static void reconnect() {
    while (!_client->connected()) {
        Serial.print("[MQTT] Connecting...");
        if (_client->connect(MQTT_CLIENT)) {
            Serial.println(" connected");
            drawStatus("MQTT OK");
            _client->subscribe(TOPIC_TTS_SPEAK);
            _client->subscribe(TOPIC_WAV_FILE);
            _client->subscribe(TOPIC_WAV_DATA);
            _client->subscribe(TOPIC_PCM_START);
            _client->subscribe(TOPIC_PCM_DATA);
            _client->subscribe(TOPIC_PCM_END);
            _client->subscribe(TOPIC_TRIGGER_REP);
            _client->subscribe(TOPIC_SCORE);
        } else {
            Serial.printf(" failed (rc=%d), retry in 3s\n", _client->state());
            drawStatus("MQTT FAILED");
            delay(3000);
        }
    }
}

// ---------------------------------------------------------------------------

void mqttSetup(WiFiClient& wifiClient) {
    static PubSubClient client(wifiClient);
    _client = &client;
    _client->setServer(MQTT_BROKER, MQTT_PORT);
    _client->setCallback(onMessage);
    _client->setBufferSize(MAX_WAV_PAYLOAD + 256);  // accommodate large WAV payloads
}

void mqttLoop() {
    if (!_client->connected()) reconnect();
    _client->loop();
}

bool mqttConnected() {
    return _client && _client->connected();
}
