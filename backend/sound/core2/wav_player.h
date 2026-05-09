#pragma once
#include <M5Core2.h>
#include <SD.h>

// Play WAV from SD card. Automatically chooses memory vs segmented based on heap.
bool playWAVFromSD(const char* filename,
                   uint32_t repeat      = 1,
                   int      channel     = -1,
                   bool     stop_current = true);

// Load entire WAV file into heap and play. Use for small files (<~100KB).
bool playWAVMemory(File&    wavFile,
                   size_t   fileSize,
                   uint32_t repeat,
                   int      channel,
                   bool     stop_current);

// Stream large WAV files from SD in chunks. Use when heap is insufficient.
bool playWAVSegmented(const char* filename,
                      uint32_t    repeat,
                      int         channel,
                      bool        stop_current);

// Play WAV from a raw in-memory buffer (data received over MQTT, etc).
bool playWAVBuffer(const uint8_t* data,
                   size_t         dataSize,
                   uint32_t       repeat      = 1,
                   int            channel     = -1,
                   bool           stop_current = true);

// Block until playback on the given channel finishes.
void waitForPlayback(int channel = SPEAKER_CHANNEL);
