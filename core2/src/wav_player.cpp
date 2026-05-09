#include "wav_player.h"
#include "config.h"

static const size_t WAV_HEADER_SIZE = 44;
static const size_t MEM_THRESHOLD   = 50 * 1024;

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
