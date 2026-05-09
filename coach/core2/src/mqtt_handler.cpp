#include "mqtt_handler.h"
#include "wav_player.h"
#include "config.h"
#include <WiFi.h>
#include <SD.h>

static PubSubClient* _client = nullptr;

extern void drawStatus(const char* msg);

static const size_t MAX_WAV_PAYLOAD = 60 * 1024;
static const size_t MAX_PCM_BUFFER = 512 * 1024;

static uint8_t* _pcmBuffer = nullptr;
static size_t _pcmSize = 0;
static size_t _pcmCapacity = 0;

static void resetPCMBuffer(size_t capacity) {
    if (_pcmBuffer) {
        free(_pcmBuffer);
        _pcmBuffer = nullptr;
    }

    _pcmSize = 0;
    _pcmCapacity = capacity;

    if (_pcmCapacity == 0) {
        return;
    }

    if (_pcmCapacity > MAX_PCM_BUFFER) {
        Serial.printf("[PCM] Requested buffer too large: %u\n", _pcmCapacity);
        drawStatus("PCM too large");
        _pcmCapacity = 0;
        return;
    }

    _pcmBuffer = (uint8_t*)malloc(_pcmCapacity);
    if (!_pcmBuffer) {
        Serial.printf("[PCM] malloc failed for %u bytes\n", _pcmCapacity);
        drawStatus("PCM no memory");
        _pcmCapacity = 0;
    }
}

// ---------------------------------------------------------------------------

static void onMessage(const char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, TOPIC_PCM_START) == 0) {
        char sizeText[16] = {0};
        size_t n = length < sizeof(sizeText) - 1 ? length : sizeof(sizeText) - 1;
        memcpy(sizeText, payload, n);
        size_t capacity = strtoul(sizeText, nullptr, 10);
        Serial.printf("[PCM] start: %u bytes\n", capacity);
        drawStatus("PCM receiving");
        resetPCMBuffer(capacity);
        return;
    }

    if (strcmp(topic, TOPIC_PCM_DATA) == 0) {
        if (!_pcmBuffer || _pcmSize + length > _pcmCapacity) {
            Serial.printf("[PCM] overflow: size=%u chunk=%u capacity=%u\n",
                          _pcmSize, length, _pcmCapacity);
            drawStatus("PCM overflow");
            return;
        }
        memcpy(_pcmBuffer + _pcmSize, payload, length);
        _pcmSize += length;
        return;
    }

    if (strcmp(topic, TOPIC_PCM_END) == 0) {
        Serial.printf("[PCM] end: %u/%u bytes\n", _pcmSize, _pcmCapacity);
        if (_pcmBuffer && _pcmSize > 0) {
            drawStatus("PCM playing");
            playPCMBuffer(_pcmBuffer, _pcmSize);
        }
        resetPCMBuffer(0);
        return;
    }

    // core2/play/file — SD filename string
    if (strcmp(topic, TOPIC_WAV_FILE) == 0) {
        char filename[64] = {0};
        size_t n = length < sizeof(filename) - 1 ? length : sizeof(filename) - 1;
        memcpy(filename, payload, n);
        Serial.printf("[MQTT] play file: %s\n", filename);
        drawStatus("MQTT play file");
        playWAVFromSD(filename);
        return;
    }

    // core2/play/data — raw WAV bytes from ElevenLabs via backend
    if (strcmp(topic, TOPIC_WAV_DATA) == 0) {
        if (length > MAX_WAV_PAYLOAD) {
            Serial.printf("[MQTT] WAV payload too large: %u\n", length);
            drawStatus("WAV too large");
            return;
        }
        Serial.printf("[MQTT] WAV data received: %u bytes\n", length);
        drawStatus("MQTT WAV received");
        playWAVBuffer((const uint8_t*)payload, length);
        return;
    }

    // core2/rep — trigger the rep-completion sound and update counter
    if (strcmp(topic, TOPIC_TRIGGER_REP) == 0) {
        bool good = (length == 4 && strncmp((char*)payload, "good", 4) == 0);
        Serial.printf("[MQTT] trigger rep good=%d\n", good);
        drawStatus(good ? "MQTT good rep" : "MQTT bad rep");
        playWAVFromSD(good ? REP_GOOD_SOUND_PATH : REP_SOUND_PATH);
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
            _client->subscribe(TOPIC_WAV_FILE);
            _client->subscribe(TOPIC_WAV_DATA);
            _client->subscribe(TOPIC_PCM_START);
            _client->subscribe(TOPIC_PCM_DATA);
            _client->subscribe(TOPIC_PCM_END);
            _client->subscribe(TOPIC_TRIGGER_REP);
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
