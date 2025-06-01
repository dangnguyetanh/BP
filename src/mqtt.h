#include <PubSubClient.h>
#include <Arduino.h>
#include <time.h>
#include <WiFi.h>
#include <time.h>
// WiFi credentials
const char* ssid = "Nguyet Anh";
const char* password = "khoa5470";

// First MQTT broker for data collection
const char* mqtt_server = "mqtt.fuvitech.vn"; 
const int mqtt_port = 1883;
const char* mqtt_topic = "Khoa/data";
const char* clientID = "ESP32";

// Second MQTT broker for risk prediction results
const char* results_mqtt_server = "app.coreiot.io"; // Replace with actual broker
const int results_mqtt_port = 1883;
const char* results_mqtt_topic = "v1/devices/me/telemetry";
const char* results_mqtt_user = "bp";        // Username for results broker
const char* results_mqtt_password = "bp";    // Password for results broker
const char* results_clientID = "bp";

// Create two separate clients
WiFiClient espClient;
PubSubClient mqttClient(espClient);

WiFiClient resultsClient;
PubSubClient resultsMqttClient(resultsClient);

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
      mqttClient.subscribe("Khoa/bp_results"); // Make sure to subscribe to BP results topic
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

void resultsMqttReconnect() 
{
  // Only try to connect if not already connected
  if (!resultsMqttClient.connected()) {
    Serial.println("Attempting connection to results MQTT broker...");
    // Connect with username and password
    if (resultsMqttClient.connect(results_clientID, results_mqtt_user, results_mqtt_password)) 
    {
      Serial.println("Connected to results MQTT broker");
    } 
    else 
    {
      Serial.print("Failed to connect to results broker, rc=");
      Serial.print(resultsMqttClient.state());
      Serial.println(" Will try again when needed");
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