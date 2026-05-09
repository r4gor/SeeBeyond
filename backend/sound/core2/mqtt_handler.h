#pragma once
#include <PubSubClient.h>

void mqttSetup(WiFiClient& wifiClient);
void mqttLoop();
bool mqttConnected();

// Called by main sketch when a display-score command arrives
extern void onScoreReceived(int score);
