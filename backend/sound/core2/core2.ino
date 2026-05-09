#include <M5Core2.h>
#include <WiFi.h>
#include <SD.h>

#include "config.h"
#include "wav_player.h"
#include "mqtt_handler.h"

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

static int currentScore = 0;
static int repCount     = 0;

static void drawScore(int score) {
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(WHITE, BLACK);

    // Large rep counter in centre
    M5.Lcd.setTextSize(6);
    M5.Lcd.setCursor(100, 60);
    M5.Lcd.printf("%d", repCount);

    // Smaller score in bottom-right
    M5.Lcd.setTextSize(3);
    M5.Lcd.setCursor(10, 200);
    M5.Lcd.setTextColor(GREEN, BLACK);
    M5.Lcd.printf("Score: %d", score);
}

static void drawStatus(const char* msg) {
    M5.Lcd.fillRect(0, 0, 320, 30, BLACK);
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(YELLOW, BLACK);
    M5.Lcd.setCursor(4, 6);
    M5.Lcd.print(msg);
}

// Called from mqtt_handler when TOPIC_SCORE arrives
void onScoreReceived(int score) {
    currentScore = score;
    drawScore(score);
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

    M5.Speaker.begin();
    M5.Speaker.setVolume(SPEAKER_VOLUME);

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

    drawScore(0);
    Serial.println("[Core2] Ready");
}

void loop() {
    M5.update();
    mqttLoop();

    // BtnA — play rep sound manually for testing
    if (M5.BtnA.wasPressed()) {
        repCount++;
        drawScore(currentScore);
        playWAVFromSD(REP_SOUND_PATH);
    }

    // BtnB — stop current playback
    if (M5.BtnB.wasPressed()) {
        M5.Speaker.stop();
    }

    // BtnC — reconnect WiFi
    if (M5.BtnC.wasPressed()) {
        wifiConnect();
    }

    delay(10);
}
