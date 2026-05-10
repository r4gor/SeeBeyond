#include <M5Core2.h>
#include <WiFi.h>
#include <SD.h>
#include <ArduinoJson.h>

#include "config.h"
#include "wav_player.h"
#include "mqtt_handler.h"

// ---------------------------------------------------------------------------
// Display zones  (screen is 320 x 240)
// ---------------------------------------------------------------------------
//
//  y=0   ┌──────────────────────────────┐
//        │  STANDING            160°    │  h=50  state bar (colored border)
//  y=50  ├──────────────────────────────┤
//        │                              │
//        │            7                 │  h=108 giant rep count
//        │                              │        bg = verdict colour
//  y=158 ├──────────────────────────────┤
//        │  + GOOD                      │  h=42  verdict line
//  y=200 ├──────────────────────────────┤
//        │  Hip 2 cm below parallel     │  h=40  feedback text
//  y=240 └──────────────────────────────┘

#define ZONE_STATE_TOP   0
#define ZONE_STATE_H     50
#define ZONE_REPS_TOP    50
#define ZONE_REPS_H      108
#define ZONE_VERD_TOP    158
#define ZONE_VERD_H      42
#define ZONE_FEED_TOP    200
#define ZONE_FEED_H      40

// ---------------------------------------------------------------------------
// Display state (updated from core2/display JSON)
// ---------------------------------------------------------------------------

static int  repCount    = 0;
static char lastVerdict [16] = "---";
static char lastFeedback[50] = "Waiting for reps...";
static char lastState   [16] = "STANDING";
static int  lastAngle   = 0;
static bool verdictGood = false;

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

static uint16_t stateBarColor() {
    if (strcmp(lastState, "DESCENDING") == 0) return M5.Lcd.color565(0,  160, 160); // teal
    if (strcmp(lastState, "BOTTOM")     == 0) return M5.Lcd.color565(220, 100,   0); // orange
    if (strcmp(lastState, "ASCENDING")  == 0) return M5.Lcd.color565(0,  200,  80); // green
    return M5.Lcd.color565(100, 100, 100);  // gray for STANDING
}

static uint16_t repsBgColor() {
    if (strcmp(lastVerdict, "---") == 0) return BLACK;
    return verdictGood ? M5.Lcd.color565(0, 55, 0)   // dark green
                       : M5.Lcd.color565(65,  0, 0); // dark red
}

// ---------------------------------------------------------------------------
// Zone draw functions
// ---------------------------------------------------------------------------

static void drawStateBar() {
    uint16_t sc  = stateBarColor();
    uint16_t bg  = M5.Lcd.color565(15, 15, 25); // very dark blue-black

    M5.Lcd.fillRect(0, ZONE_STATE_TOP, 320, ZONE_STATE_H, bg);
    M5.Lcd.drawRect(0, ZONE_STATE_TOP, 320, ZONE_STATE_H, sc);

    // State label (left)
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(sc, bg);
    M5.Lcd.setCursor(8, ZONE_STATE_TOP + 8);
    M5.Lcd.print(lastState);

    // Knee angle (right)
    char angBuf[10];
    snprintf(angBuf, sizeof(angBuf), "%d", lastAngle);
    // each char at 2x = 12px wide; append degree symbol via chr 247 (M5 font)
    int angW = (strlen(angBuf) + 1) * 12;
    M5.Lcd.setTextColor(WHITE, bg);
    M5.Lcd.setCursor(320 - angW - 8, ZONE_STATE_TOP + 8);
    M5.Lcd.print(angBuf);
    M5.Lcd.print("\xB0");  // degree symbol (0xB0 in Latin-1 / M5Stack font)
}

static void drawRepCount() {
    uint16_t bg = repsBgColor();
    M5.Lcd.fillRect(0, ZONE_REPS_TOP, 320, ZONE_REPS_H, bg);

    char buf[8];
    snprintf(buf, sizeof(buf), "%d", repCount);
    int numChars = strlen(buf);
    // At 6x: each char is 36 × 48 px
    int charW = 6 * 6;
    int charH = 8 * 6;
    int textW = numChars * charW;
    int x = (320 - textW) / 2;
    int y = ZONE_REPS_TOP + (ZONE_REPS_H - charH) / 2;

    M5.Lcd.setTextSize(6);
    M5.Lcd.setTextColor(WHITE, bg);
    M5.Lcd.setCursor(x, y);
    M5.Lcd.print(buf);
}

static void drawVerdict() {
    uint16_t bg = M5.Lcd.color565(10, 10, 10);
    M5.Lcd.fillRect(0, ZONE_VERD_TOP, 320, ZONE_VERD_H, bg);

    if (strcmp(lastVerdict, "---") == 0) {
        M5.Lcd.setTextSize(2);
        M5.Lcd.setTextColor(M5.Lcd.color565(80, 80, 80), bg);
        M5.Lcd.setCursor(8, ZONE_VERD_TOP + 12);
        M5.Lcd.print("---  no rep yet");
        return;
    }

    uint16_t tc = verdictGood ? M5.Lcd.color565(0, 220, 80)
                              : M5.Lcd.color565(220, 60, 60);
    char vBuf[32];
    // Copy + uppercase
    snprintf(vBuf, sizeof(vBuf), "%c %s", verdictGood ? '+' : '!', lastVerdict);
    for (char* p = vBuf; *p; p++) {
        if (*p >= 'a' && *p <= 'z') *p -= 32;
        if (*p == '_') *p = ' ';
    }
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(tc, bg);
    M5.Lcd.setCursor(8, ZONE_VERD_TOP + 12);
    M5.Lcd.print(vBuf);
}

static void drawFeedback() {
    uint16_t bg = M5.Lcd.color565(8, 8, 12);
    M5.Lcd.fillRect(0, ZONE_FEED_TOP, 320, ZONE_FEED_H, bg);
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(M5.Lcd.color565(180, 180, 180), bg);
    M5.Lcd.setCursor(8, ZONE_FEED_TOP + 14);
    M5.Lcd.print(lastFeedback);
}

static void drawAll() {
    drawStateBar();
    drawRepCount();
    drawVerdict();
    drawFeedback();
}

// ---------------------------------------------------------------------------
// Called by mqtt_handler when core2/display arrives
// ---------------------------------------------------------------------------

void onDisplayUpdate(const char* json) {
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, json);
    if (err) {
        Serial.printf("[JSON] parse error: %s\n", err.c_str());
        return;
    }

    repCount = doc["r"] | repCount;
    strlcpy(lastVerdict,  doc["v"] | "---",      sizeof(lastVerdict));
    strlcpy(lastFeedback, doc["f"] | "...",      sizeof(lastFeedback));
    lastAngle = doc["a"] | lastAngle;
    strlcpy(lastState,    doc["s"] | lastState,  sizeof(lastState));
    verdictGood = (strcmp(lastVerdict, "good") == 0);

    drawAll();
}

// ---------------------------------------------------------------------------
// Called by mqtt_handler when core2/rep fires (sound already played there)
// ---------------------------------------------------------------------------

void onRepReceived() {
    // Display is driven entirely by core2/display JSON — nothing to do here.
}

// ---------------------------------------------------------------------------
// Status bar helper (used during boot / WiFi / MQTT setup)
// ---------------------------------------------------------------------------

void drawStatus(const char* msg) {
    M5.Lcd.fillRect(0, ZONE_STATE_TOP, 320, ZONE_STATE_H, M5.Lcd.color565(15, 15, 25));
    M5.Lcd.setTextSize(2);
    M5.Lcd.setTextColor(YELLOW, M5.Lcd.color565(15, 15, 25));
    M5.Lcd.setCursor(4, ZONE_STATE_TOP + 8);
    M5.Lcd.print(msg);
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
    M5.begin(true, true, true, true, mbus_mode_t::kMBusModeOutput, true);
    Serial.begin(115200);
    M5.Axp.SetSpkEnable(true);
    M5.Spk.SetVolume(SPEAKER_VOLUME);

    M5.Lcd.fillScreen(BLACK);
    drawStatus("Booting...");

    if (!SD.begin(TFCARD_CS_PIN, SPI, 40000000)) {
        Serial.println("[SD] Mount failed");
        drawStatus("SD FAILED");
    } else {
        Serial.println("[SD] OK");
    }

    wifiConnect();
    mqttSetup(wifiClient);

    // Draw the initial idle layout
    drawAll();
    Serial.println("[Core2] Ready");
}

void loop() {
    M5.update();
    mqttLoop();

    // BtnA — simulate a good rep for on-stage testing
    if (M5.BtnA.wasPressed()) {
        playWAVFromSD(REP_GOOD_SOUND_PATH);
        repCount++;
        strlcpy(lastVerdict,  "good",               sizeof(lastVerdict));
        strlcpy(lastFeedback, "BtnA test rep",      sizeof(lastFeedback));
        strlcpy(lastState,    "STANDING",            sizeof(lastState));
        lastAngle   = 165;
        verdictGood = true;
        drawAll();
    }

    // BtnC — reconnect WiFi
    if (M5.BtnC.wasPressed()) {
        wifiConnect();
    }

    delay(10);
}
