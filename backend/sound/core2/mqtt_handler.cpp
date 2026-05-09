#include "mqtt_handler.h"
#include "wav_player.h"
#include "config.h"
#include <WiFi.h>
#include <SD.h>

static PubSubClient* _client = nullptr;

// Max raw WAV payload accepted over MQTT (~200 KB fits most ElevenLabs outputs)
static const size_t MAX_WAV_PAYLOAD = 200 * 1024;
static uint8_t      wavBuf[MAX_WAV_PAYLOAD];
static size_t       wavBufLen = 0;

// ---------------------------------------------------------------------------

static void onMessage(const char* topic, byte* payload, unsigned int length) {
    // core2/play/file — SD filename string
    if (strcmp(topic, TOPIC_WAV_FILE) == 0) {
        char filename[64] = {0};
        size_t n = length < sizeof(filename) - 1 ? length : sizeof(filename) - 1;
        memcpy(filename, payload, n);
        Serial.printf("[MQTT] play file: %s\n", filename);
        playWAVFromSD(filename);
        return;
    }

    // core2/play/data — raw WAV bytes from ElevenLabs via backend
    if (strcmp(topic, TOPIC_WAV_DATA) == 0) {
        if (length > MAX_WAV_PAYLOAD) {
            Serial.printf("[MQTT] WAV payload too large: %u\n", length);
            return;
        }
        Serial.printf("[MQTT] WAV data received: %u bytes\n", length);
        memcpy(wavBuf, payload, length);
        wavBufLen = length;
        playWAVBuffer(wavBuf, wavBufLen);
        return;
    }

    // core2/rep — trigger the rep-completion sound
    if (strcmp(topic, TOPIC_TRIGGER_REP) == 0) {
        Serial.println("[MQTT] trigger rep");
        playWAVFromSD(REP_SOUND_PATH);
        return;
    }

    // core2/score — display score number
    if (strcmp(topic, TOPIC_SCORE) == 0) {
        char numStr[16] = {0};
        size_t n = length < sizeof(numStr) - 1 ? length : sizeof(numStr) - 1;
        memcpy(numStr, payload, n);
        int score = atoi(numStr);
        Serial.printf("[MQTT] score: %d\n", score);
        onScoreReceived(score);
        return;
    }
}

// ---------------------------------------------------------------------------

static void reconnect() {
    while (!_client->connected()) {
        Serial.print("[MQTT] Connecting...");
        if (_client->connect(MQTT_CLIENT)) {
            Serial.println(" connected");
            _client->subscribe(TOPIC_WAV_FILE);
            _client->subscribe(TOPIC_WAV_DATA);
            _client->subscribe(TOPIC_TRIGGER_REP);
            _client->subscribe(TOPIC_SCORE);
        } else {
            Serial.printf(" failed (rc=%d), retry in 3s\n", _client->state());
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
