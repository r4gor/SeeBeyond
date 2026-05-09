#include "wav_player.h"
#include "config.h"

// WAV header offsets (standard 44-byte PCM header)
static const size_t WAV_HDR_SAMPLE_RATE = 24;  // uint32_t
static const size_t WAV_HDR_CHANNELS    = 22;  // uint16_t
static const size_t WAV_HDR_BITS        = 34;  // uint16_t
static const size_t WAV_HEADER_SIZE     = 44;

// Minimum free heap required to load a file fully into RAM
static const size_t MEM_THRESHOLD = 50 * 1024;  // 50 KB headroom

// ---------------------------------------------------------------------------

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
    } else {
        Serial.printf("[WAV] Segmented path: %s (%u bytes, heap %u free)\n", filename, fileSize, freeHeap);
        f.close();
        ok = playWAVSegmented(filename, repeat, channel, stop_current);
        return ok;
    }

    f.close();
    return ok;
}

// ---------------------------------------------------------------------------

bool playWAVMemory(File& wavFile, size_t fileSize, uint32_t repeat, int channel, bool stop_current) {
    uint8_t* buf = (uint8_t*)malloc(fileSize);
    if (!buf) {
        Serial.printf("[WAV] malloc failed for %u bytes\n", fileSize);
        return false;
    }

    wavFile.seek(0);
    size_t read = wavFile.read(buf, fileSize);
    if (read != fileSize) {
        Serial.printf("[WAV] Read mismatch: expected %u got %u\n", fileSize, read);
        free(buf);
        return false;
    }

    bool ok = M5.Speaker.playWav(buf, fileSize, repeat, channel, stop_current);
    // buf must stay alive during playback; wait then free
    waitForPlayback(channel == -1 ? SPEAKER_CHANNEL : channel);
    free(buf);
    return ok;
}

// ---------------------------------------------------------------------------

bool playWAVSegmented(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    static const size_t CHUNK = 4096;

    File f = SD.open(filename, FILE_READ);
    if (!f) return false;

    // Parse WAV header
    uint8_t hdr[WAV_HEADER_SIZE];
    if (f.read(hdr, WAV_HEADER_SIZE) != WAV_HEADER_SIZE) {
        f.close();
        return false;
    }

    uint32_t sampleRate  = *reinterpret_cast<uint32_t*>(hdr + WAV_HDR_SAMPLE_RATE);
    uint16_t numChannels = *reinterpret_cast<uint16_t*>(hdr + WAV_HDR_CHANNELS);
    uint16_t bitsPerSamp = *reinterpret_cast<uint16_t*>(hdr + WAV_HDR_BITS);
    Serial.printf("[WAV] segmented %s | %uHz %uch %ubps\n",
                  filename, sampleRate, numChannels, bitsPerSamp);

    uint8_t* chunk = (uint8_t*)malloc(CHUNK);
    if (!chunk) { f.close(); return false; }

    bool ok = true;
    for (uint32_t r = 0; r < repeat; r++) {
        if (r > 0) f.seek(WAV_HEADER_SIZE);  // rewind to audio data

        while (f.available()) {
            size_t n = f.read(chunk, CHUNK);
            if (n == 0) break;
            // wait for previous chunk to finish before queuing next
            while (M5.Speaker.isPlaying()) { delay(5); }
            if (!M5.Speaker.playWav(chunk, n, 1, channel, false)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    waitForPlayback(channel == -1 ? SPEAKER_CHANNEL : channel);
    free(chunk);
    f.close();
    return ok;
}

// ---------------------------------------------------------------------------

bool playWAVBuffer(const uint8_t* data, size_t dataSize, uint32_t repeat, int channel, bool stop_current) {
    if (!data || dataSize < WAV_HEADER_SIZE) {
        Serial.println("[WAV] Invalid buffer");
        return false;
    }

    // Copy into a mutable heap buffer that stays alive during playback
    uint8_t* buf = (uint8_t*)malloc(dataSize);
    if (!buf) {
        Serial.printf("[WAV] malloc failed for %u bytes\n", dataSize);
        return false;
    }

    memcpy(buf, data, dataSize);
    bool ok = M5.Speaker.playWav(buf, dataSize, repeat, channel, stop_current);
    waitForPlayback(channel == -1 ? SPEAKER_CHANNEL : channel);
    free(buf);
    return ok;
}

// ---------------------------------------------------------------------------

void waitForPlayback(int channel) {
    while (M5.Speaker.isPlaying()) {
        delay(10);
    }
}
