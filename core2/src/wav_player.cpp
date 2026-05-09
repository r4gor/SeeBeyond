#include "wav_player.h"
#include "config.h"

static const size_t WAV_HEADER_SIZE = 44;
static const size_t MEM_THRESHOLD   = 50 * 1024;

bool playWAVFromSD(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    if (!SD.exists(filename)) {
        Serial.printf("[WAV] File not found: %s\n", filename);
        return false;
    }

    File f = SD.open(filename, FILE_READ);
    if (!f) {
        Serial.printf("[WAV] Cannot open: %s\n", filename);
        return false;
    }

    size_t fileSize = f.size();
    size_t freeHeap = ESP.getFreeHeap();

    bool ok;
    if (freeHeap > fileSize + MEM_THRESHOLD) {
        Serial.printf("[WAV] Memory path: %s (%u bytes, heap %u free)\n", filename, fileSize, freeHeap);
        ok = playWAVMemory(f, fileSize, repeat, channel, stop_current);
        f.close();
    } else {
        Serial.printf("[WAV] Segmented path: %s (%u bytes, heap %u free)\n", filename, fileSize, freeHeap);
        f.close();
        ok = playWAVSegmented(filename, repeat, channel, stop_current);
    }
    return ok;
}

bool playWAVMemory(File& wavFile, size_t fileSize, uint32_t repeat, int channel, bool stop_current) {
    if (fileSize <= WAV_HEADER_SIZE) return false;

    uint8_t* buf = (uint8_t*)malloc(fileSize);
    if (!buf) {
        Serial.printf("[WAV] malloc failed for %u bytes\n", fileSize);
        return false;
    }

    wavFile.seek(0);
    size_t bytesRead = wavFile.read(buf, fileSize);
    if (bytesRead != fileSize) {
        Serial.printf("[WAV] Read mismatch: expected %u got %u\n", fileSize, bytesRead);
        free(buf);
        return false;
    }

    const unsigned char* pcm    = (const unsigned char*)(buf + WAV_HEADER_SIZE);
    size_t               pcmLen = fileSize - WAV_HEADER_SIZE;

    for (uint32_t r = 0; r < repeat; r++) {
        M5.Spk.PlaySound(pcm, pcmLen);
    }

    free(buf);
    return true;
}

bool playWAVSegmented(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    static const size_t CHUNK = 4096;

    uint8_t* chunk = (uint8_t*)malloc(CHUNK);
    if (!chunk) return false;

    bool ok = true;
    for (uint32_t r = 0; r < repeat; r++) {
        File f = SD.open(filename, FILE_READ);
        if (!f) { ok = false; break; }

        f.seek(WAV_HEADER_SIZE);
        while (f.available()) {
            size_t n = f.read(chunk, CHUNK);
            if (n == 0) break;
            M5.Spk.PlaySound((const unsigned char*)chunk, n);
        }
        f.close();
    }

    free(chunk);
    return ok;
}

bool playWAVBuffer(const uint8_t* data, size_t dataSize, uint32_t repeat, int channel, bool stop_current) {
    if (!data || dataSize <= WAV_HEADER_SIZE) {
        Serial.println("[WAV] Invalid buffer");
        return false;
    }

    const unsigned char* pcm    = (const unsigned char*)(data + WAV_HEADER_SIZE);
    size_t               pcmLen = dataSize - WAV_HEADER_SIZE;

    for (uint32_t r = 0; r < repeat; r++) {
        M5.Spk.PlaySound(pcm, pcmLen);
    }
    return true;
}

void waitForPlayback(int channel) {
    // PlaySound uses portMAX_DELAY — it is already synchronous, nothing to wait for
}
