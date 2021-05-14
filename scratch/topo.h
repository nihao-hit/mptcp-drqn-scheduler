#include <stdint.h>

uint32_t sub1IsSsid = 0; // 0: not init, 1: sub1 is ssid1, 2: sub1 is ssid2
uint32_t s1IsAssoc = 1; // 初始值置为1，因为当建立MPTCP连接时，两个网络都已经关联上AP了
uint32_t s2IsAssoc = 1;
uint64_t s1WifiRate = 6; // OfdmRate6Mbps是最低速率
uint64_t s2WifiRate = 6;
uint64_t s1Snr = 0;
uint64_t s2Snr = 0;