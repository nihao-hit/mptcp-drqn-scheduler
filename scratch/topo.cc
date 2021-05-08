/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
#include <vector>
#include <string>
#include <sstream>

#include "ns3/yans-wifi-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/ssid.h"
#include "ns3/object.h"
#include "ns3/node-list.h"
#include "ns3/wifi-net-device.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/v4ping-helper.h"
#include "ns3/packet-sink.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/bridge-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MptcpDrqnSchedulerTopo");

// // trace sta移动
// static void 
// CourseChange (std::string context, Ptr<const MobilityModel> mobility)
// {
//   Vector pos = mobility->GetPosition ();
//   Vector vel = mobility->GetVelocity ();
//   std::cout << Simulator::Now () << ", POS: x=" << pos.x << ",\ty=" << pos.y
//             << ";\tVEL: x=" << vel.x << ",\ty=" << vel.y << std::endl;
// }

// trace sta关联ap事件
static void
Assoc(std::string context, Mac48Address value) { 
    ////////////////////////////////////////
    // 打印关联信息，坐标
    Ptr<Node> sta = NodeList::GetNode(0);

    Ptr<MobilityModel> mobility = sta->GetObject<MobilityModel>();
    Ptr<StaWifiMac> ssid1Mac = sta->GetDevice(0)->GetObject<WifiNetDevice>()->GetMac()->GetObject<StaWifiMac>();
    Ptr<StaWifiMac> ssid2Mac = sta->GetDevice(1)->GetObject<WifiNetDevice>()->GetMac()->GetObject<StaWifiMac>();
    Vector pos = mobility->GetPosition ();
    // Vector vel = mobility->GetVelocity ();
    uint32_t ssid = atoi(context.substr (23, 1).c_str()) + 1;
    Ptr<StaWifiMac> mac = (ssid == 1) ? ssid1Mac : ssid2Mac;
    // TODO: 运行时打印出来的mac地址有点奇怪，:01有时是:01，有时是:10。
    NS_LOG_DEBUG(Simulator::Now()<<",\t  Assoc: ssid_"<<ssid<<",\tPos="<<pos<<",\tBssid="<<value);
    // 打印关联信息，坐标
    ////////////////////////////////////////

    ////////////////////////////////////////
    // 更新路由
    Ipv4StaticRoutingHelper routingHelper;
    Ptr<Ipv4> staIpv4 = sta->GetObject<Ipv4>();
    Ptr<Ipv4StaticRouting> staRouting = routingHelper.GetStaticRouting(staIpv4);

    // 删除旧route entry，包括default route
    // TODO: 删除旧route entry是否应该在DeAssoc回调中进行
    for(uint32_t rtIdx = 0; rtIdx < staRouting->GetNRoutes(); rtIdx++) {
        Ipv4RoutingTableEntry rt = staRouting->GetRoute(rtIdx);
        if(rt.GetInterface() == ssid && rt.GetGateway() != Ipv4Address("0.0.0.0")) {
            // NS_LOG_DEBUG("Deleted route entry: "<<rt);
            staRouting->RemoveRoute(rtIdx);
        }
    }

    // 添加新route entry
    // 这里AddHostRouteTo()的intf参数为什么从1开始？可能0是lo
    // ssid_1 ap: mac=[2-10], ip=10.0.1.[2-10]
    // ssid_2 ap: mac=[12-23], ip=10.0.2.[2-13]
    std::stringstream ss;
    ss<<value;
    // 注意：这里mac地址是16进制，需要转化为10进制
    std::string apMac8Str;
    while(std::getline(ss, apMac8Str, ':')){}
    int apMac8Int = std::strtol(apMac8Str.c_str(), nullptr, 16);

    std::string apIpStr = "10.0." + std::to_string(ssid) + "." +
        std::to_string(apMac8Int + ((ssid == 1) ? 0 : -10));
    Ipv4Address apIp(apIpStr.c_str());
    // NS_LOG_DEBUG("Added route entry: host="<<Ipv4Address("10.0.3.2")<<", out="<<ssid<<", next hop="<<apIp);
    staRouting->AddHostRouteTo(Ipv4Address("10.0.3.2"), apIp, ssid);
    // // 如果接口是ssid1StaDevice，添加default route
    // if(ssid == 1) {
    //     staRouting->SetDefaultRoute(apIp, ssid);
    // }
    // 更新路由
    ////////////////////////////////////////
}

// trace sta解关联ap事件
static void
DeAssoc(std::string context, Mac48Address value) {
    Ptr<Node> sta = NodeList::GetNode(0);
    Ptr<MobilityModel> mobility = sta->GetObject<MobilityModel>();
    Vector pos = mobility->GetPosition ();
    uint32_t ssid = atoi(context.substr (23, 1).c_str()) + 1;
    NS_LOG_DEBUG(Simulator::Now()<<",\tDeAssoc: ssid_"<<ssid<<",\tPos="<<pos<<",\tBssid="<<value);
}

// trace sta接口rate
static void
Rate(std::string context, uint64_t oldValue, uint64_t newValue){
    uint32_t ssid = atoi(context.substr (23, 1).c_str()) + 1;
    NS_LOG_DEBUG(Simulator::Now()<<"ssid_"<<ssid<<",\toldRate="<<oldValue<<",\tnewRate"<<newValue);
}

int main(int argc, char *argv[])
{
    ////////////////////////////////////////////////////////////////////////////////
    // 命令行参数及全局配置
    LogComponentEnable("MptcpDrqnSchedulerTopo", LOG_DEBUG);
    // LogComponentEnable("MpTcpSocketBase", LOG_DEBUG);

    // Set the maximum wireless range to 30 meters
    Config::SetDefault("ns3::RangePropagationLossModel::MaxRange", DoubleValue(30));

    // TODO: 不知道为什么，ssid1与ssid2在每一秒都会发生一次漫游，不论如何降低速度。
    // 配置移动模型
    Config::SetDefault ("ns3::RandomWalk2dMobilityModel::Mode", StringValue ("Time"));
    Config::SetDefault ("ns3::RandomWalk2dMobilityModel::Time", StringValue ("15s"));
    Config::SetDefault ("ns3::RandomWalk2dMobilityModel::Speed", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=2.0]"));
    Config::SetDefault ("ns3::RandomWalk2dMobilityModel::Bounds", StringValue ("0|120|0|120"));
    
    // 配置mptcp
    Config::SetDefault("ns3::MpTcpSocketBase::Epoch", TimeValue(MilliSeconds(200)));
    Config::SetDefault("ns3::MpTcpSocketBase::RewardAlpha", DoubleValue(0.3));
    Config::SetDefault("ns3::MpTcpSocketBase::RewardBeta", DoubleValue(0.5));
    Config::SetDefault("ns3::MpTcpSocketBase::LstmSeqLen", UintegerValue(8));

    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(0));
    Config::SetDefault("ns3::DropTailQueue::Mode", StringValue("QUEUE_MODE_PACKETS"));
    Config::SetDefault("ns3::DropTailQueue::MaxPackets", UintegerValue(100));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(MpTcpSocketBase::GetTypeId()));
    Config::SetDefault("ns3::MpTcpSocketBase::MaxSubflows", UintegerValue(8)); // Sink
    // 命令行参数及全局配置
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 创建节点
    Ptr<Node> sta = CreateObject<Node>();
    Ptr<Node> s1 = CreateObject<Node>();
    Ptr<Node> s2 = CreateObject<Node>();
    Ptr<Node> router = CreateObject<Node>();
    Ptr<Node> server = CreateObject<Node>();
    NodeContainer ssid1ApNodes;
    ssid1ApNodes.Create(9);
    NodeContainer ssid2ApNodes;
    ssid2ApNodes.Create(12);
    
    NodeContainer lan1Nodes;
    lan1Nodes.Add(sta);
    lan1Nodes.Add(ssid1ApNodes);
    lan1Nodes.Add(s1);
    lan1Nodes.Add(router);
    NodeContainer lan2Nodes;
    lan2Nodes.Add(sta);
    lan2Nodes.Add(ssid2ApNodes);
    lan2Nodes.Add(s2);
    lan2Nodes.Add(router);
    NodeContainer lan3Nodes;
    lan3Nodes.Add(router);
    lan3Nodes.Add(server);
    // 创建节点
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 初始化ssid1网络，为sta, ssid1Aps附加wifi接口
    ////////////////////////////////////////
    // 初始化wifi helper, mobility helper
    NetDeviceContainer ssid1StaDevice;
    NetDeviceContainer ssid1ApWifiDevices;
    // TODO: sta在离开当前关联ap的范围才会断开连接
    // Create a channel helper and phy helper, and then create the channel
    YansWifiChannelHelper channel1 = YansWifiChannelHelper::Default();
    channel1.AddPropagationLoss("ns3::RangePropagationLossModel");
    YansWifiPhyHelper phy1 = YansWifiPhyHelper::Default();
    phy1.SetPcapDataLinkType(YansWifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy1.SetChannel(channel1.Create());
    // Create a WifiMacHelper, which is reused across STA and AP configurations
    NqosWifiMacHelper mac1 = NqosWifiMacHelper::Default();
    // Create a WifiHelper, which will use the above helpers to create
    // and install Wifi devices.  Configure a Wifi standard to use, which
    // will align various parameters in the Phy and Mac to standard defaults.
    WifiHelper wifi1 = WifiHelper::Default();
    // Configure mobility
    MobilityHelper mobility1;
    Ssid ssid1 = Ssid("ssid_1");
    // 初始化wifi helper, mobility helper
    ////////////////////////////////////////

    ////////////////////////////////////////
    // 为sta添加ssid1 sta wifi device
    // Perform the installation
    mac1.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid1),
                "ActiveProbing", BooleanValue(false));
    ssid1StaDevice = wifi1.Install(phy1, mac1, sta);
    ////////////////////////////////////////
    
    ////////////////////////////////////////
    // 为ssid1Aps添加ap wifi device，设置mobility model
    mobility1.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility1.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(30.0),
                                  "MinY", DoubleValue(30.0),
                                  "DeltaX", DoubleValue(30.0),
                                  "DeltaY", DoubleValue(30.0),
                                  "GridWidth", UintegerValue(3),
                                  "LayoutType", StringValue("RowFirst"));
    for (uint32_t i = 0; i < ssid1ApNodes.GetN(); i++)
    {
        // Perform the installation
        mac1.SetType("ns3::ApWifiMac",
                    "Ssid", SsidValue(ssid1));
        ssid1ApWifiDevices.Add(wifi1.Install(phy1, mac1, ssid1ApNodes.Get(i)));
        // Configure mobility
        mobility1.Install(ssid1ApNodes.Get(i));
    }
    // 为ssid1Aps添加ap wifi device，设置mobility model
    ////////////////////////////////////////
    // 初始化ssid1网络，为sta, ssid1Aps附加wifi接口
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 初始化ssid2网络，为sta, ssid2Aps附加wifi接口
    ////////////////////////////////////////
    // 初始化wifi helper, mobility helper
    NetDeviceContainer ssid2StaDevice;
    NetDeviceContainer ssid2ApWifiDevices;
    // TODO: 现在两个ssid都使用的默认802.11a，是否需要配置mac协议异构性
    // Create a channel helper and phy helper, and then create the channel
    YansWifiChannelHelper channel2 = YansWifiChannelHelper::Default();
    channel2.AddPropagationLoss("ns3::RangePropagationLossModel");
    YansWifiPhyHelper phy2 = YansWifiPhyHelper::Default();
    phy2.SetPcapDataLinkType(YansWifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy2.SetChannel(channel2.Create());
    // Create a WifiMacHelper, which is reused across STA and AP configurations
    NqosWifiMacHelper mac2 = NqosWifiMacHelper::Default();
    // Create a WifiHelper, which will use the above helpers to create
    // and install Wifi devices.  Configure a Wifi standard to use, which
    // will align various parameters in the Phy and Mac to standard defaults.
    WifiHelper wifi2 = WifiHelper::Default();
    // Configure mobility
    MobilityHelper mobility2;
    Ssid ssid2 = Ssid("ssid_2");
    // 初始化wifi helper, mobility helper
    ////////////////////////////////////////

    ////////////////////////////////////////
    // 为sta添加ssid2 sta wifi device
    // Perform the installation
    mac2.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid2),
                "ActiveProbing", BooleanValue(false));
    ssid2StaDevice = wifi2.Install(phy2, mac2, sta);
    
    // 为sta设置移动模型
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(60.0, 60.0, 0.0));
    mobility2.SetPositionAllocator(positionAlloc);
    mobility2.SetMobilityModel("ns3::RandomWalk2dMobilityModel");
    mobility2.Install(sta);
    ////////////////////////////////////////

    ////////////////////////////////////////
    // 为ssid2Aps添加ap wifi device，设置mobility model
    mobility2.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(15.0),
                                  "MinY", DoubleValue(30.0),
                                  "DeltaX", DoubleValue(30.0),
                                  "DeltaY", DoubleValue(30.0),
                                  "GridWidth", UintegerValue(4),
                                  "LayoutType", StringValue("RowFirst"));
    for (uint32_t i = 0; i < ssid2ApNodes.GetN(); i++)
    {
        // Perform the installation
        mac2.SetType("ns3::ApWifiMac",
                    "Ssid", SsidValue(ssid2));
        ssid2ApWifiDevices.Add(wifi2.Install(phy2, mac2, ssid2ApNodes.Get(i)));
        // Configure mobility
        mobility2.Install(ssid2ApNodes.Get(i));
    }
    // 为ssid2Aps添加ap wifi device，设置mobility model
    ////////////////////////////////////////
    // 初始化ssid2网络，为sta, ssid2Aps附加wifi接口
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 为节点附加csma接口
    // 方便同一局域网创建IP地址
    NetDeviceContainer lan1Devices;
    NetDeviceContainer lan2Devices;
    NetDeviceContainer lan3Devices;
    // 方便同一设备设置接口间桥接
    NetDeviceContainer ssid1ApCsmaDevices;
    NetDeviceContainer ssid2ApCsmaDevices;
    NetDeviceContainer s1Devices;
    NetDeviceContainer s2Devices;

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));
    
    NetDeviceContainer link;

    // 附加lan1 device (wifi, csma)
    lan1Devices.Add(ssid1StaDevice);
    lan1Devices.Add(ssid1ApWifiDevices);
    for (uint32_t i = 0; i < ssid1ApNodes.GetN(); i++)
    {
        link = csma.Install(NodeContainer(ssid1ApNodes.Get(i), s1));
        ssid1ApCsmaDevices.Add(link.Get(0));
        s1Devices.Add(link.Get(1));
        lan1Devices.Add(link);
    }
    link = csma.Install(NodeContainer(s1, router));
    s1Devices.Add(link.Get(0));
    lan1Devices.Add(link);

    // 附加lan2 device (wifi, csma)
    lan2Devices.Add(ssid2StaDevice);
    lan2Devices.Add(ssid2ApWifiDevices);
    for (uint32_t i = 0; i < ssid2ApNodes.GetN(); i++)
    {
        link = csma.Install(NodeContainer(ssid2ApNodes.Get(i), s2));
        ssid2ApCsmaDevices.Add(link.Get(0));
        s2Devices.Add(link.Get(1));
        lan2Devices.Add(link);
    }
    link = csma.Install(NodeContainer(s2, router));
    s2Devices.Add(link.Get(0));
    lan2Devices.Add(link);

    // 附加lan3 device (csma)
    lan3Devices.Add(csma.Install(NodeContainer(router, server)));
    // 为节点附加csma接口
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 设置AP和SWITCH接口间桥接
    //
    // 示例说明：
    // host1<--csma-->switch<--csma-->host2，如果不为switch配置Bridge，host1 ping host2会失败，
    // 抓包发现对于host1发出的arp请求，switch的左边接口并不转发给右边接口。
    // 注意：必须为创建的Bridge Device分配IP地址，否则host1 ping switch左右接口都会失败。
    //
    BridgeHelper bridge;
    NetDeviceContainer lan1BridgeDevs;
    NetDeviceContainer lan2BridgeDevs;

    // 设置ssid1 ap bridge
    for(uint32_t i = 0; i < ssid1ApNodes.GetN(); i++) 
    {
        NetDeviceContainer tmp;
        tmp.Add(ssid1ApWifiDevices.Get(i));
        tmp.Add(ssid1ApCsmaDevices.Get(i));
        lan1BridgeDevs.Add(bridge.Install(ssid1ApNodes.Get(i), tmp));
    }

    // 设置ssid2 ap bridge
    for(uint32_t i = 0; i < ssid2ApNodes.GetN(); i++) 
    {
        NetDeviceContainer tmp;
        tmp.Add(ssid2ApWifiDevices.Get(i));
        tmp.Add(ssid2ApCsmaDevices.Get(i));
        lan2BridgeDevs.Add(bridge.Install(ssid2ApNodes.Get(i), tmp));
    }

    // 设置switch bridge
    lan1BridgeDevs.Add(bridge.Install(s1, s1Devices));
    lan2BridgeDevs.Add(bridge.Install(s2, s2Devices));
    // 设置AP和SWITCH接口间桥接
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 为所有节点添加TCP/IP协议栈
    InternetStackHelper stack;
    stack.Install(sta);
    stack.Install(ssid1ApNodes);
    stack.Install(s1);
    stack.Install(ssid2ApNodes);
    stack.Install(s2);
    stack.Install(router);
    stack.Install(server);

    Ipv4AddressHelper address;
    address.SetBase("10.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer lan1Intfs = address.Assign(lan1Devices);
    address.Assign(lan1BridgeDevs);

    address.SetBase("10.0.2.0", "255.255.255.0");
    Ipv4InterfaceContainer lan2Intfs = address.Assign(lan2Devices);
    address.Assign(lan2BridgeDevs);

    address.SetBase("10.0.3.0", "255.255.255.0");
    Ipv4InterfaceContainer lan3Intfs = address.Assign(lan3Devices);

    // 打印所有节点信息
    for(uint32_t nodeIdx = 0; nodeIdx < NodeList::GetNNodes(); nodeIdx++) {
        Ptr<Node> node = NodeList::GetNode(nodeIdx);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        for(uint32_t intfIdx = 0; intfIdx < ipv4->GetNInterfaces(); intfIdx++) {
            for(uint32_t addrIdx = 0; addrIdx < ipv4->GetNAddresses(intfIdx); addrIdx++) {
                Ipv4InterfaceAddress ip = ipv4->GetAddress(intfIdx, addrIdx);
                NS_LOG_INFO("nodeIdx="<<nodeIdx<<", intfIdx="<<intfIdx
                                      <<", addrIdx="<<addrIdx<<", ip="<<ip);
            }
        }
        for(uint32_t devIdx = 0; devIdx < node->GetNDevices(); devIdx++) {
            Ptr<NetDevice> dev = node->GetDevice(devIdx);
            NS_LOG_INFO("nodeIdx="<<nodeIdx<<", ifIdx="<<dev->GetIfIndex()
                                  <<", mac address="<<dev->GetAddress());
        }
        // ipv4->GetNInterfaces()：intfIdx：Interface number of an Ipv4 interface
        // dev->GetIfIndex()：ifIdx：index ifIndex of the device
    }
    // 为所有节点添加TCP/IP协议栈
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 设置路由，注意：所有节点都需要设置路由
    Ipv4StaticRoutingHelper routingHelper;
    Ptr<Ipv4> s1Ipv4 = s1->GetObject<Ipv4>();
    Ptr<Ipv4> s2Ipv4 = s2->GetObject<Ipv4>();
    Ptr<Ipv4> routerIpv4 = router->GetObject<Ipv4>();
    Ptr<Ipv4> serverIpv4 = server->GetObject<Ipv4>();

    Ptr<Ipv4StaticRouting> s1Routing = routingHelper.GetStaticRouting(s1Ipv4);
    Ptr<Ipv4StaticRouting> s2Routing = routingHelper.GetStaticRouting(s2Ipv4);
    Ptr<Ipv4StaticRouting> routerRouting = routingHelper.GetStaticRouting(routerIpv4);
    Ptr<Ipv4StaticRouting> serverRouting = routingHelper.GetStaticRouting(serverIpv4);

    s1Routing->AddNetworkRouteTo(Ipv4Address("10.0.3.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.1.30"), 10);

    s2Routing->AddNetworkRouteTo(Ipv4Address("10.0.3.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.2.39"), 13);

    routerRouting->AddNetworkRouteTo(Ipv4Address("10.0.1.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.1.29"), 1);
    routerRouting->AddNetworkRouteTo(Ipv4Address("10.0.2.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.2.38"), 2);

    serverRouting->AddNetworkRouteTo(Ipv4Address("10.0.1.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.3.1"), 1);
    serverRouting->AddNetworkRouteTo(Ipv4Address("10.0.2.0"), Ipv4Mask("255.255.255.0"), Ipv4Address("10.0.3.1"), 1);

    for(uint32_t i = 0; i < ssid1ApNodes.GetN(); i++) {
        Ptr<Node> ap = ssid1ApNodes.Get(i);
        Ptr<Ipv4> ipv4 = ap->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> routing = routingHelper.GetStaticRouting(ipv4);
        // ipAddr从intf=lo,idx=0递增
        uint32_t ipAddr = ipv4->GetAddress(2, 0).GetLocal().Get();
        // 初始化ap<->switch的csma链路，因此接口ip递增
        routing->AddNetworkRouteTo(Ipv4Address("10.0.3.0"), Ipv4Mask("255.255.255.0"), Ipv4Address(ipAddr + 1), 2);
    }
    for(uint32_t i = 0; i < ssid2ApNodes.GetN(); i++) {
        Ptr<Node> ap = ssid2ApNodes.Get(i);
        Ptr<Ipv4> ipv4 = ap->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> routing = routingHelper.GetStaticRouting(ipv4);
        uint32_t ipAddr = ipv4->GetAddress(2, 0).GetLocal().Get();
        // 初始化ap<->switch的csma链路，因此接口ip递增
        routing->AddNetworkRouteTo(Ipv4Address("10.0.3.0"), Ipv4Mask("255.255.255.0"), Ipv4Address(ipAddr + 1), 2);
    }

    // 打印所有节点的路由表	 
    // Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper>(&std::cout);
    // routingHelper.PrintRoutingTableAllEvery(Seconds(10), routingStream);
    // 设置路由，注意：所有节点都需要设置路由
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // 设置应用层流量
    // sta ping server
    // V4PingHelper ping ("10.0.3.2");
    // ping.SetAttribute ("Interval", TimeValue (Seconds(1.0)));
    // ping.SetAttribute ("Size", UintegerValue (1024));
    // ping.SetAttribute ("Verbose", BooleanValue (true));

    // ApplicationContainer apps = ping.Install (sta);
    // apps.Start (Seconds (1.0));
    // apps.Stop (Seconds (100.0));

    // mptcp
    uint16_t port = 9;
    // 注意：BulkSendApplication默认置MaxBytes=0,即尽可能多，尽可能快的发包
    MpTcpBulkSendHelper sendHelper("ns3::TcpSocketFactory", InetSocketAddress("10.0.3.2", port));
    ApplicationContainer sendApp = sendHelper.Install(sta);

    // 注意：测试发现ns3执行sta依次关联ssid1,ssid2基站事件分别发生在<1ms，1.5ms时刻，
    // 而mptcp只在主流三次握手期间尝试建立所有子流，因此延迟mptcp流量发送事件。
    sendApp.Start(Seconds(5.0));
    sendApp.Stop(Seconds(10.0));

    MpTcpPacketSinkHelper rcvHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer rcvApp = rcvHelper.Install(server);

    rcvApp.Start(Seconds(4.0));
    rcvApp.Stop(Seconds(11.0));
    // 设置应用层流量
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // trace
    // Config::Connect ("/NodeList/0/$ns3::MobilityModel/CourseChange", MakeCallback (&CourseChange));
    
    Config::Connect("/NodeList/0/DeviceList/[0-1]/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/Assoc", MakeCallback(&Assoc));
    Config::Connect("/NodeList/0/DeviceList/[0-1]/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/DeAssoc", MakeCallback(&DeAssoc));

    // TODO: 为ArfWifiManager添加Rate trace
    Config::Connect("/NodeList/0/DeviceList/[0-1]/$ns3::WifiNetDevice/RemoteStationManager/$ns3::ArfWifiManager/Rate", MakeCallback(&Rate));

    phy1.EnablePcap("MptcpDrqnSchedulerTopo", ssid1StaDevice);
    phy2.EnablePcap("MptcpDrqnSchedulerTopo", ssid2StaDevice);
    csma.EnablePcap("MptcpDrqnSchedulerTopo", lan3Devices.Get(1));
    // trace
    ////////////////////////////////////////////////////////////////////////////////

    Simulator::Stop(Seconds(15.0));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
