#include <M5Core2.h>
#include <WiFi.h>
#include <SD.h>

#include "config.h"
#include "wav_player.h"
#include "mqtt_handler.h"

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

static int repCount = 0;

static void drawReps() {
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(WHITE, BLACK);
    M5.Lcd.setTextSize(6);
    M5.Lcd.setCursor(100, 60);
    M5.Lcd.printf("%d", repCount);
}

static void drawStatus(const char* msg) {
    M5.Lcd.fillRect(0, 0, 320, 30, BLACK);
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(YELLOW, BLACK);
    M5.Lcd.setCursor(4, 6);
    M5.Lcd.print(msg);
}

void onRepReceived() {
    repCount++;
    drawReps();
}

// ---------------------------------------------------------------------------
// WiFi
// ---------------------------------------------------------------------------

static void wifiConnect() {
    drawStatus("WiFi connecting...");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
        drawStatus("WiFi OK");
    } else {
        Serial.println("[WiFi] Failed");
        drawStatus("WiFi FAILED");
    }
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------

WiFiClient wifiClient;

void setup() {
    M5.begin();
    Serial.begin(115200);

    M5.Spk.begin();

    M5.Lcd.fillScreen(BLACK);
    drawStatus("Booting...");

    if (!SD.begin()) {
        Serial.println("[SD] Mount failed");
        drawStatus("SD FAILED");
    } else {
        Serial.println("[SD] OK");
    }

    wifiConnect();

    mqttSetup(wifiClient);

    drawReps();
    Serial.println("[Core2] Ready");
}

void loop() {
    M5.update();
    mqttLoop();

    // BtnA — simulate rep for testing
    if (M5.BtnA.wasPressed()) {
        playWAVFromSD(REP_SOUND_PATH);
        onRepReceived();
    }

    // BtnC — reconnect WiFi
    if (M5.BtnC.wasPressed()) {
        wifiConnect();
    }

    delay(10);
}
