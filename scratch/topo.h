#include <stdint.h>

uint32_t sub1IsSsid = 0; // 0: not init, 1: sub1 is ssid1, 2: sub1 is ssid2

double posX = 60;
double posY = 60;
uint32_t ssid1IsAssoc = 0;
uint32_t ssid2IsAssoc = 0;
uint64_t ssid1WifiRate = 6; // OfdmRate6Mbps是最低速率
uint64_t ssid2WifiRate = 6;
uint64_t ssid1Snr = 0;
uint64_t ssid2Snr = 0;