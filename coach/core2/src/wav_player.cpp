#include "wav_player.h"
#include "config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const size_t WAV_HEADER_SIZE = 44;
static const size_t MEM_THRESHOLD   = 50 * 1024;

// ---------------------------------------------------------------------------
// Streaming audio task
//
// Each pcm/data MQTT chunk is enqueued immediately and played as it arrives,
// so playback starts after the first chunk instead of waiting for all of them.
// All audio (PCM stream + SD rep sounds) goes through this single task so
// M5.Spk.PlaySound is never called concurrently from two contexts.
// ---------------------------------------------------------------------------

// Chunks to buffer before playback starts. Each MQTT chunk is typically a few
// KB; 3 chunks gives ~100-150ms of runway so the network can stay ahead of the
// speaker without adding noticeable latency.
static const int MIN_PREBUFFER = 3;

struct AudioChunk { uint8_t* data; size_t size; };
static QueueHandle_t    _audioQueue = nullptr;
static volatile bool    _buffering  = false;
static volatile int     _chunkCount = 0;

static void audioTask(void*) {
    AudioChunk chunk;
    for (;;) {
        // Hold here while pre-buffering so the queue can fill before playback.
        while (_buffering) vTaskDelay(pdMS_TO_TICKS(10));
        if (xQueueReceive(_audioQueue, &chunk, pdMS_TO_TICKS(100)) == pdTRUE) {
            M5.Spk.PlaySound((const unsigned char*)chunk.data, chunk.size);
            free(chunk.data);
        }
    }
}

void initAudioTask() {
    _audioQueue = xQueueCreate(8, sizeof(AudioChunk));
    xTaskCreate(audioTask, "audio", 8192, nullptr, 2, nullptr);
}

// Enqueue raw PCM. Takes ownership of `data` (will free it after playback).
bool queueAudioChunk(uint8_t* data, size_t size) {
    if (!data || size == 0 || !_audioQueue) { free(data); return false; }
    AudioChunk chunk = {data, size};
    if (xQueueSend(_audioQueue, &chunk, pdMS_TO_TICKS(500)) != pdTRUE) {
        free(data);
        Serial.println("[Audio] queue full, chunk dropped");
        return false;
    }
    if (_buffering && ++_chunkCount >= MIN_PREBUFFER) {
        _buffering = false;
        Serial.println("[Audio] pre-buffer full, starting playback");
    }
    return true;
}

// Begin a new PCM stream: flush stale audio and enable pre-buffering.
void startAudioStream() {
    if (!_audioQueue) return;
    AudioChunk chunk;
    while (xQueueReceive(_audioQueue, &chunk, 0) == pdTRUE) {
        if (chunk.data) free(chunk.data);
    }
    _chunkCount = 0;
    _buffering  = true;
}

// End a PCM stream: release any remaining buffered chunks for playback even if
// MIN_PREBUFFER was not reached (short audio cue case).
void endAudioStream() {
    _buffering = false;
}

// Discard all pending chunks. Called on pcm/start to flush any stale audio.
void flushAudioQueue() {
    if (!_audioQueue) return;
    AudioChunk chunk;
    while (xQueueReceive(_audioQueue, &chunk, 0) == pdTRUE) {
        if (chunk.data) free(chunk.data);
    }
}

extern void drawStatus(const char* msg);

struct __attribute__((packed)) WavHeader {
    char riff[4];
    uint32_t chunkSize;
    char wave[4];
    char fmt[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t channels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

struct __attribute__((packed)) WavSubChunk {
    char id[4];
    uint32_t size;
};

static bool findPCMData(File& f) {
    f.seek(0);

    WavHeader header{};
    if (f.read((uint8_t*)&header, sizeof(header)) != sizeof(header)) {
        Serial.println("[WAV] Header read failed");
        drawStatus("WAV bad header");
        return false;
    }

    if (memcmp(header.riff, "RIFF", 4) || memcmp(header.wave, "WAVE", 4) ||
        memcmp(header.fmt, "fmt ", 4) || header.audioFormat != 1 ||
        header.channels != 1 || header.sampleRate != 44100 ||
        header.bitsPerSample != 16) {
        Serial.printf("[WAV] Unsupported format: rate=%u bits=%u channels=%u format=%u\n",
                      header.sampleRate, header.bitsPerSample, header.channels,
                      header.audioFormat);
        drawStatus("WAV bad format");
        return false;
    }

    WavSubChunk chunk{};
    while (f.read((uint8_t*)&chunk, sizeof(chunk)) == sizeof(chunk)) {
        if (memcmp(chunk.id, "data", 4) == 0) {
            return true;
        }
        f.seek(f.position() + chunk.size);
    }

    Serial.println("[WAV] data chunk not found");
    drawStatus("WAV no data");
    return false;
}

static const uint8_t* findPCMData(const uint8_t* data, size_t dataSize, size_t& pcmLen) {
    if (!data || dataSize < sizeof(WavHeader) + sizeof(WavSubChunk)) {
        Serial.println("[WAV] Invalid buffer");
        drawStatus("WAV invalid");
        return nullptr;
    }

    const WavHeader* header = (const WavHeader*)data;
    if (memcmp(header->riff, "RIFF", 4) || memcmp(header->wave, "WAVE", 4) ||
        memcmp(header->fmt, "fmt ", 4) || header->audioFormat != 1 ||
        header->channels != 1 || header->sampleRate != 44100 ||
        header->bitsPerSample != 16) {
        Serial.printf("[WAV] Unsupported buffer format: rate=%u bits=%u channels=%u format=%u\n",
                      header->sampleRate, header->bitsPerSample, header->channels,
                      header->audioFormat);
        drawStatus("WAV bad format");
        return nullptr;
    }

    size_t offset = sizeof(WavHeader);
    while (offset + sizeof(WavSubChunk) <= dataSize) {
        const WavSubChunk* chunk = (const WavSubChunk*)(data + offset);
        offset += sizeof(WavSubChunk);
        if (memcmp(chunk->id, "data", 4) == 0) {
            if (offset + chunk->size > dataSize) {
                Serial.println("[WAV] Truncated data chunk");
                drawStatus("WAV truncated");
                return nullptr;
            }
            pcmLen = chunk->size;
            return data + offset;
        }
        offset += chunk->size;
    }

    Serial.println("[WAV] data chunk not found");
    drawStatus("WAV no data");
    return nullptr;
}

// Extract PCM from an in-memory WAV buffer, enqueue for playback.
bool queueWAVBuffer(const uint8_t* data, size_t size) {
    size_t pcmLen = 0;
    const uint8_t* pcm = findPCMData(data, size, pcmLen);
    if (!pcm) return false;
    uint8_t* copy = (uint8_t*)malloc(pcmLen);
    if (!copy) { Serial.println("[Audio] malloc failed for WAV buffer"); return false; }
    memcpy(copy, pcm, pcmLen);
    return queueAudioChunk(copy, pcmLen);
}

// Load a WAV from SD, extract PCM, enqueue for playback.
bool queueWAVFromSD(const char* filename) {
    if (!SD.exists(filename)) {
        Serial.printf("[Audio] SD file missing: %s\n", filename);
        return false;
    }
    File f = SD.open(filename, FILE_READ);
    if (!f) { Serial.printf("[Audio] cannot open: %s\n", filename); return false; }

    size_t fileSize = f.size();
    uint8_t* buf = (uint8_t*)malloc(fileSize);
    if (!buf) { f.close(); Serial.println("[Audio] malloc failed for SD WAV"); return false; }
    f.read(buf, fileSize);
    f.close();

    size_t pcmLen = 0;
    const uint8_t* pcm = findPCMData(buf, fileSize, pcmLen);
    if (!pcm) { free(buf); return false; }

    uint8_t* pcmCopy = (uint8_t*)malloc(pcmLen);
    if (!pcmCopy) { free(buf); Serial.println("[Audio] malloc failed for PCM"); return false; }
    memcpy(pcmCopy, pcm, pcmLen);
    free(buf);

    return queueAudioChunk(pcmCopy, pcmLen);
}

bool playWAVFromSD(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    if (!SD.exists(filename)) {
        Serial.printf("[WAV] File not found: %s\n", filename);
        drawStatus("WAV file missing");
        return false;
    }

    File f = SD.open(filename, FILE_READ);
    if (!f) {
        Serial.printf("[WAV] Cannot open: %s\n", filename);
        drawStatus("WAV open failed");
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
        drawStatus("WAV no memory");
        return false;
    }

    wavFile.seek(0);
    size_t bytesRead = wavFile.read(buf, fileSize);
    if (bytesRead != fileSize) {
        Serial.printf("[WAV] Read mismatch: expected %u got %u\n", fileSize, bytesRead);
        free(buf);
        drawStatus("WAV read failed");
        return false;
    }

    size_t pcmLen = 0;
    const uint8_t* pcm = findPCMData(buf, fileSize, pcmLen);
    if (!pcm) {
        free(buf);
        return false;
    }

    for (uint32_t r = 0; r < repeat; r++) {
        M5.Spk.PlaySound((const unsigned char*)pcm, pcmLen);
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

        if (!findPCMData(f)) {
            f.close();
            ok = false;
            break;
        }
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
    size_t pcmLen = 0;
    const uint8_t* pcm = findPCMData(data, dataSize, pcmLen);
    if (!pcm) {
        return false;
    }

    for (uint32_t r = 0; r < repeat; r++) {
        M5.Spk.PlaySound((const unsigned char*)pcm, pcmLen);
    }
    return true;
}

bool playPCMBuffer(const uint8_t* data, size_t dataSize, uint32_t repeat) {
    if (!data || dataSize == 0) {
        Serial.println("[PCM] Invalid buffer");
        drawStatus("PCM invalid");
        return false;
    }

    for (uint32_t r = 0; r < repeat; r++) {
        M5.Spk.PlaySound((const unsigned char*)data, dataSize);
    }
    return true;
}

void waitForPlayback(int channel) {
    // PlaySound uses portMAX_DELAY — it is already synchronous, nothing to wait for
}
