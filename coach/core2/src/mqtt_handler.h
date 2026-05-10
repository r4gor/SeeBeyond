#pragma once
#include <WiFi.h>
#include <PubSubClient.h>

void mqttSetup(WiFiClient& wifiClient);
void mqttLoop();
bool mqttConnected();

extern void onRepReceived();
extern void onDisplayUpdate(const char* json);
