#include "tts_client.h"
#include "wav_player.h"
#include "config.h"

#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

extern void drawStatus(const char* msg);

static QueueHandle_t _ttsQueue = nullptr;

struct TTSRequest { char text[256]; };

// ---------------------------------------------------------------------------

static void jsonEscape(const char* src, char* dst, size_t dstLen) {
    size_t i = 0, j = 0;
    while (src[i] && j + 3 < dstLen) {
        char c = src[i++];
        if (c == '"' || c == '\\') dst[j++] = '\\';
        dst[j++] = c;
    }
    dst[j] = '\0';
}

static void ttsTask(void*) {
    TTSRequest req;
    for (;;) {
        if (xQueueReceive(_ttsQueue, &req, portMAX_DELAY) != pdTRUE) continue;

        Serial.printf("[TTS] %s\n", req.text);
        drawStatus("TTS fetching");

        char escaped[300];
        jsonEscape(req.text, escaped, sizeof(escaped));

        char body[400];
        snprintf(body, sizeof(body),
            "{\"text\":\"%s\",\"model_id\":\"" ELEVENLABS_MODEL_ID "\"}",
            escaped);

        WiFiClientSecure sec;
        sec.setInsecure();
        HTTPClient http;
        http.begin(sec,
            "https://api.elevenlabs.io/v1/text-to-speech/" ELEVENLABS_VOICE_ID
            "?output_format=mp3_22050_32");
        http.addHeader("Content-Type", "application/json");
        http.addHeader("xi-api-key", ELEVENLABS_API_KEY);
        http.setTimeout(12000);

        int code = http.POST((uint8_t*)body, strlen(body));
        if (code != 200) {
            String resp = http.getString();
            Serial.printf("[TTS] HTTP %d: %s\n", code, resp.c_str());
            char errMsg[32];
            snprintf(errMsg, sizeof(errMsg), "TTS err %d", code);
            drawStatus(errMsg);
            http.end();
            continue;
        }

        static constexpr size_t kMaxPCMBytes = 512 * 1024; // 512 KB PSRAM cap
        int contentLen = http.getSize();
        size_t capacity = (contentLen > 0 && (size_t)contentLen <= kMaxPCMBytes)
                          ? (size_t)contentLen
                          : kMaxPCMBytes;

        uint8_t* buf = (uint8_t*)ps_malloc(capacity);
        if (!buf) {
            Serial.println("[TTS] ps_malloc failed");
            drawStatus("TTS no mem");
            http.end();
            continue;
        }

        WiFiClient* stream = http.getStreamPtr();
        size_t received = 0;
        uint32_t deadline = millis() + 14000;

        while ((http.connected() || stream->available()) && received < capacity) {
            if (millis() > deadline) { Serial.println("[TTS] timeout"); break; }
            size_t avail = stream->available();
            if (avail == 0) { delay(1); continue; }
            received += stream->readBytes(buf + received,
                                          min(avail, capacity - received));
        }
        http.end();

        Serial.printf("[TTS] %u bytes\n", received);
        if (received > 0) {
            drawStatus("TTS playing");
            queueAudioChunk(buf, received);
        } else {
            free(buf);
            drawStatus("TTS empty");
        }
    }
}

// ---------------------------------------------------------------------------

void initTTSTask() {
    _ttsQueue = xQueueCreate(1, sizeof(TTSRequest));
    xTaskCreate(ttsTask, "tts", 20480, nullptr, 2, nullptr);
}

bool scheduleTTS(const char* text) {
    if (!_ttsQueue) return false;
    TTSRequest req;
    strncpy(req.text, text, sizeof(req.text) - 1);
    req.text[sizeof(req.text) - 1] = '\0';
    if (xQueueSend(_ttsQueue, &req, 0) != pdTRUE) {
        Serial.println("[TTS] queue full, dropped");
        return false;
    }
    return true;
}
