#pragma once
#include <M5Core2.h>
#include <SD.h>
#include "config.h"

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

bool playPCMBuffer(const uint8_t* data,
                   size_t         dataSize,
                   uint32_t       repeat       = 1);

// Block until playback on the given channel finishes.
void waitForPlayback(int channel = SPEAKER_CHANNEL);

// Start the background audio task (call once from setup()).
void initAudioTask();

// Enqueue raw 44100 Hz / 16-bit / mono PCM for streaming playback.
// Takes ownership of `data` — the task frees it after playing.
bool queueAudioChunk(uint8_t* data, size_t size);

// Begin a new PCM stream: flush stale audio and enable pre-buffering.
void startAudioStream();

// Signal end of stream so buffered chunks drain even if MIN_PREBUFFER not reached.
void endAudioStream();

// Discard all pending chunks in the queue.
void flushAudioQueue();

// Extract PCM from a WAV buffer and enqueue it.
bool queueWAVBuffer(const uint8_t* data, size_t size);

// Load a WAV file from SD, extract PCM, and enqueue it.
bool queueWAVFromSD(const char* filename);
