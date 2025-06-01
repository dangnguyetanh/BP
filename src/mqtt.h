#include <PubsubClient.h>
#include <Arduino.h>
#include <time.h>
#include <WiFi.h>
#include <time.h>
// WiFi credentials
const char* ssid = "Nguyet Anh";
const char* password = "khoa5470";

const char* mqtt_server = "mqtt.fuvitech.vn"; 
const int mqtt_port = 1883;
const char* mqtt_topic = "Khoa/data";
// const char* mqtt_user = "bp";           // Username MQTT
// const char* mqtt_password = "bp";  // Password MQTT
const char* clientID = "bp";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

void connectWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void mqttReconnect() 
{
  while (!mqttClient.connected()) {
    Serial.println("Attempting MQTT connection...");
    if (mqttClient.connect(clientID)) 
    {
      Serial.println("Connected to MQTT broker");
      mqttClient.subscribe(mqtt_topic);
    } 
    else 
    {
      Serial.print("Failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" Retrying in 5 seconds...");
      delay(5000);
    }
  }
}

String getTimestamp() {
  time_t now;
  struct tm timeinfo;
  time(&now);
  localtime_r(&now, &timeinfo);
  char timeStr[20];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(timeStr);
}