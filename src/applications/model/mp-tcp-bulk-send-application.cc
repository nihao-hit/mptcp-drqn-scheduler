 /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Sussex (UK)
 * Copyright (c) 2010 Georgia Institute of Technology
 *
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
 *
 * Author: George F. Riley <riley@ece.gatech.edu>
 * Modified by Morteza Kheikhah <m.kheirkhah@sussex.ac.uk>
 */

#include "ns3/json.hpp"

#include "ns3/log.h"
#include "ns3/address.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/tcp-socket-factory.h"
#include "mp-tcp-bulk-send-application.h"
#include "ns3/string.h"

#include "ns3/random-variable-stream.h"
#include "scratch/topo.h"

using json = nlohmann::ordered_json;

NS_LOG_COMPONENT_DEFINE ("MpTcpBulkSendApplication");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (MpTcpBulkSendApplication)
  ;

TypeId
MpTcpBulkSendApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MpTcpBulkSendApplication")
    .SetParent<Application> ()
    .AddConstructor<MpTcpBulkSendApplication> ()
    .AddAttribute ("SendSize", "The amount of data to send each time from application buffer to socket buffer.",
                   UintegerValue (140000), //512
                   MakeUintegerAccessor (&MpTcpBulkSendApplication::m_sendSize),
                   MakeUintegerChecker<uint32_t> (1))
//    .AddAttribute ("BufferSize", "The size of the application buffer.",
//                    UintegerValue(10),
//                    MakeUintegerAccessor(&MpTcpBulkSendApplication::m_bufferSize),
//                    MakeUintegerChecker<uint32_t>(1))
    .AddAttribute ("Remote", "The address of the destination",
                   AddressValue (),
                   MakeAddressAccessor (&MpTcpBulkSendApplication::m_peer),
                   MakeAddressChecker ())
    .AddAttribute ("MaxBytes",
                   "The total number of bytes to send. "
                   "Once these bytes are sent, "
                   "no data  is sent again. The value zero means "
                   "that there is no limit.",
                   UintegerValue (0), // 1 MB default of data to send
                   MakeUintegerAccessor (&MpTcpBulkSendApplication::m_maxBytes),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute("FlowId",
                  "Unique Id for each flow installed per node",
                  UintegerValue(0),
                  MakeUintegerAccessor (&MpTcpBulkSendApplication::m_flowId),
                  MakeUintegerChecker<uint32_t> ())
    .AddAttribute("MaxSubflows",
                  "Number of MPTCP subflows",
                  UintegerValue(8),
                  MakeUintegerAccessor (&MpTcpBulkSendApplication::m_subflows),
                  MakeUintegerChecker<uint32_t> ())
    .AddAttribute("SimTime",
                  "Simulation Time for this application it should be smaller than stop time",
                   UintegerValue(20),
                   MakeUintegerAccessor(&MpTcpBulkSendApplication::m_simTime),
                   MakeUintegerChecker<uint32_t>())
    .AddAttribute("DupAck",
                 "Dupack threshold -- used only for MMPTCP and PacketScatter",
                  UintegerValue(0),
                  MakeUintegerAccessor(&MpTcpBulkSendApplication::m_dupack),
                  MakeUintegerChecker<uint32_t>())
    .AddAttribute("FlowType",
                  "The Type of the Flow: Large or Short ",
                  StringValue("Large"),
                  MakeStringAccessor(&MpTcpBulkSendApplication::m_flowType),
                  MakeStringChecker())
    .AddAttribute("OutputFileName",
                  "Output file name",
                  StringValue("NULL"),
                  MakeStringAccessor(&MpTcpBulkSendApplication::m_outputFileName),
                  MakeStringChecker())
    .AddAttribute ("Protocol", "The type of protocol to use.",
                   TypeIdValue (TcpSocketFactory::GetTypeId ()),
                   MakeTypeIdAccessor (&MpTcpBulkSendApplication::m_tid),
                   MakeTypeIdChecker ())
    .AddTraceSource ("Tx", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&MpTcpBulkSendApplication::m_txTrace))
    .AddAttribute ("DataIntervalUpper", "",
          TimeValue (MilliSeconds(200)),
          MakeTimeAccessor (&MpTcpBulkSendApplication::dataIntervalUpper),
          MakeTimeChecker())
    .AddAttribute ("DataIntervalLower", "",
          TimeValue (MilliSeconds(100)),
          MakeTimeAccessor (&MpTcpBulkSendApplication::dataIntervalLower),
          MakeTimeChecker())
  ;
  return tid;
}


MpTcpBulkSendApplication::MpTcpBulkSendApplication ()
  : m_socket (0),
    m_connected (false),
    m_totBytes (0)
{
  NS_LOG_FUNCTION (this);
//  m_data = new uint8_t[m_bufferSize];
  //memset(m_data, 0, m_sendSize*2);
}

MpTcpBulkSendApplication::~MpTcpBulkSendApplication ()
{
  NS_LOG_FUNCTION (this);
//  delete [] m_data;
//  m_data = 0;
}

void
MpTcpBulkSendApplication::SetBuffer(uint32_t buffSize){
  NS_LOG_FUNCTION_NOARGS();
//  delete [] m_data;
//  m_data = 0;
//  m_data = new uint8_t[buffSize];
  //memset(m_data, 0, m_bufferSize);
}

void
MpTcpBulkSendApplication::SetMaxBytes (uint32_t maxBytes)
{
  NS_LOG_FUNCTION (this << maxBytes);
  m_maxBytes = maxBytes;
}

Ptr<Socket>
MpTcpBulkSendApplication::GetSocket (void) const
{
  NS_LOG_FUNCTION (this);
  return m_socket;
}

void
MpTcpBulkSendApplication::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  m_socket = 0;
  Application::DoDispose (); // chain up
}

// Application Methods
void MpTcpBulkSendApplication::StartApplication (void) // Called at time specified by Start
{
  NS_LOG_FUNCTION (this);
  //NS_LOG_UNCOND(Simulator::Now().GetSeconds() << " StartApplication -> Node-FlowId: {" << GetNode()->GetId() <<"-" << m_flowId<< "} MaxBytes: " << m_maxBytes << " F-Type: " << m_flowType << " S-Time: " << m_simTime);
  // Create the socket if not already
  if (!m_socket)
    {
      //m_socket = CreateObject<MpTcpSocketBase>(GetNode()); //m_socket = Socket::CreateSocket (GetNode (), m_tid);
      m_socket = DynamicCast<MpTcpSocketBase>(Socket::CreateSocket (GetNode (), m_tid));
      m_socket->Bind();
      //m_socket->SetMaxSubFlowNumber(m_subflows);
      m_socket->SetFlowType(m_flowType);
      m_socket->SetOutputFileName(m_outputFileName);
      int result = m_socket->Connect(m_peer);
      if (result == 0)
        {
          m_socket->SetFlowId(m_flowId);
          m_socket->SetDupAckThresh(m_dupack);
          m_socket->SetConnectCallback(MakeCallback(&MpTcpBulkSendApplication::ConnectionSucceeded, this),
              MakeCallback(&MpTcpBulkSendApplication::ConnectionFailed, this));
          // cxxx: 只保留ConnectionSucceeded()->SendData()一个入口
          // m_socket->SetDataSentCallback(MakeCallback(&MpTcpBulkSendApplication::DataSend, this));
          m_socket->SetCloseCallbacks (
            MakeCallback (&MpTcpBulkSendApplication::HandlePeerClose, this),
            MakeCallback (&MpTcpBulkSendApplication::HandlePeerError, this));

          //m_socket->SetSendCallback(MakeCallback(&MpTcpBulkSendApplication::DataSend, this));
        }
      else
        {
          NS_LOG_UNCOND("Connection is failed");
        }
    }
  if (m_connected) // cxxx: 这里条件理论上不可能成立
    {
      SendData ();
    }
}

// tuple traceRtt -> nlohmann json
void to_json(json& j, const tuple<int64_t, uint32_t, int64_t, int64_t>& t) {
  j = json{
    {"timestamp", get<0>(t)},
    {"localAddr", get<1>(t)},
    {"rtt", get<2>(t)},
    {"srtt", get<3>(t)}
  };
}

// nlohmann json -> tuple traceRtt
void from_json(const json& j, tuple<int64_t, uint32_t, int64_t, int64_t>& t) {
  auto& timestamp = j.at("timestamp");
  auto& localAddr = j.at("localAddr");
  auto& rtt = j.at("rtt");
  auto& srtt = j.at("srtt");
  t = make_tuple(timestamp, localAddr, rtt, srtt);
}

// tuple traceTx -> nlohmann json
void to_json(json& j, const tuple<int64_t, uint32_t, string, uint64_t, 
                                  uint16_t, uint32_t, uint32_t, uint32_t>& t) {
  j = json{
    {"timestamp", get<0>(t)},
    {"localAddr", get<1>(t)},
    {"type", get<2>(t)},
    {"dataSeq", get<3>(t)},
    {"dataLen", get<4>(t)},
    {"subSeq", get<5>(t)},
    {"subAck", get<6>(t)},
    {"dupAckCnt", get<7>(t)}
  };
}

// nlohmann json -> tuple traceTx
void from_json(const json& j, tuple<int64_t, uint32_t, string, uint64_t, 
                                    uint16_t, uint32_t, uint32_t, uint32_t>& t) {
  auto& timestamp = j.at("timestamp");
  auto& localAddr = j.at("localAddr");
  auto& type = j.at("type");
  auto& dataSeq = j.at("dataSeq");
  auto& dataLen = j.at("dataLen");
  auto& subSeq = j.at("subSeq");
  auto& subAck = j.at("subAck");
  auto& dupAckCnt = j.at("dupAckCnt");
  t = make_tuple(timestamp, localAddr, type, dataSeq, dataLen, subSeq, subAck, dupAckCnt);
}

void MpTcpBulkSendApplication::StopApplication (void) // Called at time specified by Stop
{
  NS_LOG_FUNCTION (this);
  NS_LOG_UNCOND(Simulator::Now().GetSeconds() << " ["<<m_node->GetId() << "] Application STOP");

  // cxxx: 与StartApplication()前后呼应，避免采集垃圾时间的经验元组
  if(m_socket->epochId.IsRunning()){
    m_socket->epochId.Cancel();
  }

  // cxxx: 退出时必须取消所有事件，否则仿真会被阻塞
  if(sendDataEvent.IsRunning()){
    sendDataEvent.Cancel();
  }

  // 调度算法性能trace保存
  string scheName;
  switch(m_socket->GetDataDistribAlgo()) {
    case Round_Robin:
      scheName = "RoundRobin";
      break;
    case DRQN:
      scheName = "DRQN";
      break;
    case MinRtt:
      scheName = "MinRtt";
      break;
    case Redundant:
      scheName = "Redundant";
      break;
  }
  string path = "/home/cx/Desktop";

  json traceRttJson(m_socket->traceRtt);
  ofstream traceRttOf(path + "/" + scheName + "-traceRtt.json");
  traceRttOf<<traceRttJson.dump()<<endl;

  json traceTxJson(m_socket->traceTx);
  ofstream traceTxOf(path + "/" + scheName + "-traceTx.json");
  traceTxOf<<traceTxJson.dump()<<endl;

  if (m_socket != 0)
    {
      m_socket->Close ();
      m_connected = false;
    }
  else
    {
      NS_LOG_WARN ("MpTcpBulkSendApplication found null socket to close in StopApplication");
    }
}

void
MpTcpBulkSendApplication::HandlePeerClose (Ptr<Socket> socket)
{
  //StopApplication();
  NS_LOG_UNCOND("*** ["<< m_node->GetId() << "] HandlePeerError is called -> connection is false");
  m_connected = false;
  NS_LOG_FUNCTION (this << socket);
}

void
MpTcpBulkSendApplication::HandlePeerError (Ptr<Socket> socket)
{
  //StopApplication();
  NS_LOG_UNCOND("*** ["<< m_node->GetId() << "] HandlePeerError is called -> connection is false");
  m_connected = false;
  NS_LOG_FUNCTION (this << socket);
}

// Private helpers
void MpTcpBulkSendApplication::SendData (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG("m_totBytes: " << m_totBytes << " maxByte: " << m_maxBytes << " GetTxAvailable: " << m_socket->GetTxAvailable() << " SendSize: " << m_sendSize);

  uint32_t toSend = m_sendSize;
  if (m_maxBytes > 0)
    {
      toSend = std::min(m_sendSize, m_maxBytes - m_totBytes);
    }
  //while (m_totBytes < m_maxBytes && m_socket->GetTxAvailable())
  while(toSend != 0)
    { // Time to send more new data into MPTCP socket buffer
          //toSend = std::min(toSend, m_bufferSize);
          //int actual = m_socket->FillBuffer(&m_data[toSend], toSend); // TODO Change m_totalBytes to toSend
          int actual = m_socket->FillBuffer(toSend); // TODO Change m_totalBytes to toSend
          m_totBytes += actual;
          toSend -= actual;
          NS_LOG_DEBUG("toSend: " << toSend << " actual: " << actual << " totalByte: " << m_totBytes);
          m_socket->SendBufferedData();
    }
  if (m_totBytes == m_maxBytes && m_connected)
    {
      m_socket->Close();
      m_connected = false;
    }
  if(m_connected) {
    static Ptr<UniformRandomVariable> random = CreateObject<UniformRandomVariable> ();
    uint32_t dataInterval = random->GetInteger(dataIntervalLower.GetInteger(), 
                                               dataIntervalUpper.GetInteger());
    sendDataEvent = Simulator::Schedule(Time(dataInterval), &MpTcpBulkSendApplication::SendData, this);
  }
}

void MpTcpBulkSendApplication::ConnectionSucceeded (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_LOG_LOGIC ("MpTcpBulkSendApplication Connection succeeded");
  m_connected = true;

  // cxxx: 确认ssid与subflow对应关系
  if(m_socket->m_localAddress == Ipv4Address("10.0.1.1")) {
    sub1IsSsid = 1;
  } else {
    sub1IsSsid = 2;
  }
  // cxxx: 在应用启动连接建立后调度经验元组采集函数
  m_socket->epochId = Simulator::ScheduleNow(&MpTcpSocketBase::scheduleEpoch, m_socket);

  // cxxx: 只保留该SendData()一个入口，关闭DataSend()回调入口，StartApplication()入口理论上不可能成立
  //
  // 如果DataSend()回调入口存在，一次SendData()可能触发多次SendData()
  // SendData()->SendBufferedData()->SendPendingData()->NotifyDataSent()->DataSend()->SendData()
  //        ReceivedAck()->NewAck()->
  //                     ->DupAck()->
  //                 ProcessSynSent->
  SendData ();
}

void MpTcpBulkSendApplication::ConnectionFailed (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_LOG_LOGIC ("MpTcpBulkSendApplication, Connection Failed");
}

void MpTcpBulkSendApplication::DataSend (Ptr<Socket>, uint32_t)
{
  NS_LOG_FUNCTION (this);
  if (m_connected)
    { // Only send new data if the connection has completed
      Simulator::ScheduleNow(&MpTcpBulkSendApplication::SendData, this);
    }
}

} // Namespace ns3
