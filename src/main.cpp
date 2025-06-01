#include <Arduino.h>
#include <Wire.h>
#include <M5AtomS3.h>
#include "process.h"
// #include "bp.h"
#include "MAX30100_PulseOximeter.h"
#include "mqtt.h"
#include <ArduinoJson.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // Thay micro_log.h
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>

// Add these global variables for MQTT subscription
float SBP_value = 0;
float DBP_value = 0;
bool bp_data_received = false;
String riskResult = "";
// Cấu hình mô hình
namespace {
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroAllocator* allocator = nullptr;
}  // namespace

// Thông số chuẩn hóa (cập nhật từ scaler_mean.npy và scaler_scale.npy)
float scaler_mean[4] = {79.53374663,97.50437243, 124.4379712, 79.49962504  };
float scaler_scale[4] = { 11.55286498, 1.44259433,8.65692398, 5.75723374 };

// Biến và hằng số cho cảm biến MAX30100
PulseOximeter pox;
#define REPORTING_PERIOD_MS 1000
uint32_t tsLastReport = 0;
float heartRate = 0;
int spO2 = 0;
bool spo2Ready = false;
bool fingerDetected = false;

// Biến đếm và theo dõi cho việc xác định giá trị cuối cùng
#define STABLE_COUNT_REQUIRED 5  // Số lần đo ổn định cần thiết
#define MAX_ATTEMPTS 20          // Số lần đo tối đa trước khi bị coi là thất bại
int measurementCount = 0;        // Số lần đo đã thực hiện
int stableCount = 0;             // Số lần đo liên tiếp với giá trị ổn định
float lastHeartRate = 0;         // Giá trị HR cuối cùng để so sánh
int lastSpO2 = 0;                // Giá trị SpO2 cuối cùng để so sánh
float finalHeartRate = 0;        // Giá trị HR cuối cùng đã xác định
int finalSpO2 = 0;               // Giá trị SpO2 cuối cùng đã xác định

// Enum trạng thái chương trình
enum ProgramState {
  WAITING_FOR_FINGER,
  MEASURING_HR_SPO2,
  AWAITING_BP_START,
  MEASURING_BP,
  COMPLETED
};

ProgramState programState = WAITING_FOR_FINGER;

// Add this function to perform the health risk prediction
void predictHealthRisk(float heartRate, int spO2, float sbp, float dbp) {
  // Normalize the input data based on the training scaler
  float normalized_hr = (heartRate - scaler_mean[0]) / scaler_scale[0];
  float normalized_spo2 = (spO2 - scaler_mean[1]) / scaler_scale[1];
  float normalized_sbp = (sbp - scaler_mean[2]) / scaler_scale[2];
  float normalized_dbp = (dbp - scaler_mean[3]) / scaler_scale[3];
  
  // Set input tensor values
  input->data.f[0] = normalized_hr;
  input->data.f[1] = normalized_spo2;
  input->data.f[2] = normalized_sbp;
  input->data.f[3] = normalized_dbp;
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    riskResult = "Error";
    return;
  }
  
  // Get the output and determine risk level
  float risk_score = output->data.f[0];
  Serial.print("Risk score: ");
  Serial.println(risk_score);
  
  if (risk_score > 0.5) {
    riskResult = "High Risk";
  } else {
    riskResult = "Low Risk";
  }
  
  Serial.print("Risk prediction: ");
  Serial.println(riskResult);
  
  // Get current timestamp
  String timestamp = getTimestamp();
  
  if (resultsMqttClient.connected()) {
    // Create the JSON document for the results
    DynamicJsonDocument resultsDoc(256);
    resultsDoc["systolic"] = round(sbp * 10) / 10.0;    // Round to 1 decimal place
    resultsDoc["diastolic"] = round(dbp * 10) / 10.0;   // Round to 1 decimal place
    resultsDoc["heart_rate"] = round(heartRate * 10) / 10.0;
    resultsDoc["spo2"] = spO2;
    resultsDoc["risk"] = riskResult;
    resultsDoc["timestamp"] = timestamp;
    
    String resultsJson;
    serializeJson(resultsDoc, resultsJson);
    
    // Publish to the results MQTT broker
    bool published = resultsMqttClient.publish(results_mqtt_topic, resultsJson.c_str());
    
    if (published) {
      Serial.println("Published results to external MQTT broker:");
      Serial.println(resultsJson);
    } else {
      Serial.println("Failed to publish results to external broker");
    }
  } else {
    Serial.println("Could not connect to external MQTT broker for results");
  }
}
// Add this callback function for MQTT message reception
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  // Create a buffer for the payload
  char message[length + 1];
  for (int i = 0; i < length; i++) {
    message[i] = (char)payload[i];
  }
  message[length] = '\0';
  Serial.println(message);

  // Process the BP results from MQTT
  if (String(topic) == "Khoa/bp_results") {
    DynamicJsonDocument doc(256);
    DeserializationError error = deserializeJson(doc, message);
    
    if (!error) {
      SBP_value = doc["SBP"];
      DBP_value = doc["DBP"];
      bp_data_received = true;
      
      Serial.print("Received BP values - SBP: ");
      Serial.print(SBP_value);
      Serial.print(", DBP: ");
      Serial.println(DBP_value);
      
      // Predict risk once BP data is received
      if (programState == MEASURING_BP && bp_data_received && finalHeartRate > 0 && finalSpO2 > 0) {
        predictHealthRisk(finalHeartRate, finalSpO2, SBP_value, DBP_value);
        programState = COMPLETED;
      }
    } else {
      Serial.print("deserializeJson() failed: ");
      Serial.println(error.c_str());
    }
  }
}



void onBeatDetected() {
  Serial.println("Beat detected!");
  if (programState == WAITING_FOR_FINGER) {
    fingerDetected = true;
    programState = MEASURING_HR_SPO2;

    // Reset các biến đếm và theo dõi
    measurementCount = 0;
    stableCount = 0;
    lastHeartRate = 0;
    lastSpO2 = 0;

    // Hiển thị thông báo đang đo SpO2 và Heart Rate
    M5.Display.fillScreen(BLACK);
    M5.Display.setTextColor(WHITE);
    M5.Display.setTextSize(2);
    M5.Display.setTextDatum(middle_center);
    M5.Display.drawString("MEASURING", 64, 40);
    M5.Display.drawString("SPO2 & HR", 64, 64);
  }
}

// Hàm hiển thị thông báo lên màn hình
void displayMessage(const char* messages[], int lines, int textSize = 2, int x = 64, int y = 64) {
  M5.Display.fillScreen(BLACK);
  M5.Display.setTextColor(WHITE);
  M5.Display.setTextSize(textSize);
  M5.Display.setTextDatum(middle_center);
  int lineHeight = textSize * 12;
  int startY = y - (lineHeight * (lines - 1) / 2);
  for (int i = 0; i < lines; i++) {
    M5.Display.drawString(messages[i], x, startY + i * lineHeight);
  }
}

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load mô hình
  model = tflite::GetModel(health_risk_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version mismatch!");
    return;
  }

  // Thiết lập OpResolver
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();

  // Thiết lập MicroAllocator
  allocator = tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize, error_reporter);

  // Khởi tạo interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, allocator, error_reporter);
  interpreter = &static_interpreter;

  // Cấp phát bộ nhớ
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Lấy tensor input và output
  input = interpreter->input(0);
  output = interpreter->output(0);
  M5.begin();
  Serial.begin(115200);
  Wire.begin(2, 1);

  connectWiFi();
  
  // Set up the first MQTT client
  mqttClient.setServer(mqtt_server, mqtt_port);
  mqttClient.setCallback(mqttCallback);
  mqttReconnect();
  resultsMqttReconnect();
  mqttClient.subscribe("Khoa/bp_results");
  mqttClient.publish(mqtt_topic, "hello from ESP32");
  
  // Set up the second MQTT client for results
  resultsMqttClient.setServer(results_mqtt_server, results_mqtt_port);
  
  // Configure NTP for real-time
  configTime(7 * 3600, 0, "pool.ntp.org");  // UTC+7 for Vietnam
  while (!time(nullptr)) {
    Serial.println("Waiting for NTP time sync...");
    delay(1000);
  }
  Serial.println("Time synchronized");

  // Khởi tạo cảm biến MAX30100
  Serial.print("Initializing pulse oximeter..");
  if (!pox.begin()) {
    Serial.println("FAILED");
    const char* errorMsg[] = { "SENSOR", "ERROR" };
    displayMessage(errorMsg, 2);
    while (1) {
      delay(100);
    }
  } else {
    Serial.println("SUCCESS");
  }

  // Cấu hình cảm biến MAX30100
  pox.setOnBeatDetectedCallback(onBeatDetected);
  pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);

  // Khởi tạo cảm biến ADS1115
  if (!ads.begin(0x48)) {
    Serial.println("ADS init failed");
    const char* errorMsg[] = { "ADS SENSOR", "ERROR" };
    displayMessage(errorMsg, 2);
    while (1) {
      delay(100);
    }
  }

  // Cấu hình pins cho bơm hơi và van
  pinMode(SLOW_RELEASE_PIN, OUTPUT);
  pinMode(PUMP_PIN0, OUTPUT);
  pinMode(PUMP_PIN1, OUTPUT);
  pinMode(TURN_OFF_PIN0, OUTPUT);
  pinMode(TURN_OFF_PIN1, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  state = IDLE;
  offPump();
  onSlowRelease();
  onTurnOff();
  Serial.println("Setup done");

  Serial.print("Free heap memory: ");
  Serial.println(ESP.getFreeHeap());

  // Hiển thị thông báo ban đầu
  const char* startMsg[] = { "PUT FINGER", "ON SENSOR" };
  displayMessage(startMsg, 2);

  Serial.println("Place finger on sensor to measure SpO2 and Heart Rate");
}

void loop() {
  if (!mqttClient.connected()) mqttReconnect();
  mqttClient.loop();
  if (!resultsMqttClient.connected()) resultsMqttReconnect();
  resultsMqttClient.loop();
  unsigned long currentMillis = millis();
  
  M5.update();
  pox.update();

  switch (programState) {
    case WAITING_FOR_FINGER:
      // Thêm vào phần này để kiểm tra tín hiệu từ MAX30100
      if (currentMillis - tsLastReport > 500) {  // Kiểm tra mỗi 500ms
        tsLastReport = currentMillis;

        // Sử dụng getHeartRate để kiểm tra xem cảm biến đã bắt đầu nhận được tín hiệu chưa
        heartRate = pox.getHeartRate();
        spO2 = pox.getSpO2();

        // Nếu phát hiện có giá trị hợp lệ (không phải 0)
        if (heartRate > 0 || spO2 > 0) {
          fingerDetected = true;
          programState = MEASURING_HR_SPO2;

          // Reset các biến đếm và theo dõi
          measurementCount = 0;
          stableCount = 0;
          lastHeartRate = 0;
          lastSpO2 = 0;

          // Hiển thị thông báo đang đo SpO2 và Heart Rate
          M5.Display.fillScreen(BLACK);
          M5.Display.setTextColor(WHITE);
          M5.Display.setTextSize(2);
          M5.Display.setTextDatum(middle_center);
          M5.Display.drawString("MEASURING", 64, 40);
          M5.Display.drawString("SPO2 & HR", 64, 64);

          Serial.println("Finger detected through HR/SpO2 values!");
        }

        // Hiển thị trạng thái chờ đợi ngón tay
        M5.Display.fillScreen(BLACK);
        M5.Display.setTextColor(WHITE);
        M5.Display.setTextSize(2);
        M5.Display.drawString("PUT FINGER", 64, 40);
        M5.Display.drawString("ON SENSOR", 64, 70);

        // Hiển thị giá trị để debug (tùy chọn)
        M5.Display.setTextSize(1);
        char debugStr[40];
        sprintf(debugStr, "HR: %.1f, SpO2: %d", heartRate, spO2);
        M5.Display.drawString(debugStr, 64, 100);
      }

      // Giữ mã kiểm tra nút nhấn để bypass trạng thái chờ ngón tay nếu cần
      if (digitalRead(BUTTON_PIN) == LOW) {
        while (digitalRead(BUTTON_PIN) == LOW) {}  // Chờ thả nút
        // Chuyển sang đo huyết áp luôn (nếu muốn bypass cả đo SpO2 và HR)
        finalHeartRate = 70.0;  // Giá trị mặc định
        finalSpO2 = 95;         // Giá trị mặc định
        spo2Ready = true;
        programState = AWAITING_BP_START;

        const char* skipMsg[] = { "PRESS BUTTON", "FOR BP MEASURE" };
        displayMessage(skipMsg, 2);
      }
      break;

    case MEASURING_HR_SPO2:
      // Đo SpO2 và nhịp tim
      if (currentMillis - tsLastReport > REPORTING_PERIOD_MS) {
        tsLastReport = currentMillis;
        measurementCount++;

        heartRate = pox.getHeartRate();
        spO2 = pox.getSpO2();

        Serial.print("Heart rate: ");
        Serial.print(heartRate);
        Serial.print(" bpm / SpO2: ");
        Serial.print(spO2);
        Serial.print(" % | Attempt: ");
        Serial.print(measurementCount);
        Serial.print(" | Stable count: ");
        Serial.println(stableCount);

        // Hiển thị giá trị đang đo lên màn hình
        M5.Display.fillScreen(BLACK);
        M5.Display.setTextColor(WHITE);
        M5.Display.setTextSize(2);

        char hrStr[20], spo2Str[20], countStr[20];
        sprintf(hrStr, "HR: %.1f", heartRate);
        sprintf(spo2Str, "SpO2: %d%%", spO2);
        sprintf(countStr, "Stable: %d/5", stableCount);

        M5.Display.setTextDatum(middle_center);
        M5.Display.drawString("MEASURING", 64, 20);
        M5.Display.drawString(hrStr, 64, 50);
        M5.Display.drawString(spo2Str, 64, 80);
        M5.Display.setTextSize(1);
        M5.Display.drawString(countStr, 64, 110);

        // Kiểm tra giá trị ổn định
        if (heartRate >= 50 && spO2 >= 70) {
          // Kiểm tra xem giá trị có giống với lần đo trước không
          // Cho phép sai số SpO2 là 2 và Heart Rate là 5
          if (abs(heartRate - lastHeartRate) <= 5.0 && abs(spO2 - lastSpO2) <= 2) {
            stableCount++;
            Serial.print("Stable values detected! Count: ");
            Serial.println(stableCount);
          } else {
            stableCount = 1;  // Reset về 1 (không phải 0) vì giá trị hiện tại vẫn hợp lệ
          }

          // Cập nhật giá trị cuối cùng để so sánh lần sau
          lastHeartRate = heartRate;
          lastSpO2 = spO2;

          // Nếu đủ số lần ổn định, xác định đây là giá trị cuối cùng
          if (stableCount >= STABLE_COUNT_REQUIRED) {
            finalHeartRate = heartRate;
            finalSpO2 = spO2;
            spo2Ready = true;

            Serial.println("SpO2 and Heart Rate measurement complete!");
            Serial.print("Final HR: ");
            Serial.print(finalHeartRate);
            Serial.print(" bpm, Final SpO2: ");
            Serial.print(finalSpO2);
            Serial.println("%");
            Serial.println("Press button to start blood pressure measurement");

            // Hiển thị kết quả đo và thông báo sẵn sàng đo BP
            M5.Display.fillScreen(BLACK);
            M5.Display.setTextSize(2);

            char finalHrStr[20], finalSpo2Str[20];
            sprintf(finalHrStr, "HR: %.1f", finalHeartRate);
            sprintf(finalSpo2Str, "SpO2: %d%%", finalSpO2);

            M5.Display.drawString("COMPLETED", 64, 30);
            M5.Display.drawString(finalHrStr, 64, 60);
            M5.Display.drawString(finalSpo2Str, 64, 90);
            M5.Display.setTextSize(1);
            M5.Display.drawString("PRESS BUTTON", 64, 120);

            programState = AWAITING_BP_START;
          }
        } else {
          stableCount = 0;
        }

      }

      // Kiểm tra nút nhấn để bỏ qua đo SpO2 (trường hợp khẩn cấp)
      if (digitalRead(BUTTON_PIN) == LOW) {
        while (digitalRead(BUTTON_PIN) == LOW) {}  // Chờ thả nút
        spo2Ready = true;                          // Giả lập đã đo xong

        // Nếu chưa có giá trị ổn định, sử dụng giá trị cuối cùng đo được
        if (stableCount < STABLE_COUNT_REQUIRED) {
          finalHeartRate = (heartRate >= 50) ? heartRate : 70.0;  // Giá trị mặc định nếu không hợp lệ
          finalSpO2 = (spO2 >= 70) ? spO2 : 95;                   // Giá trị mặc định nếu không hợp lệ
        }

        Serial.println("Skipping SpO2 measurement");
        Serial.println("Press button again to start blood pressure measurement");

        const char* skipMsg[] = { "PRESS BUTTON", "FOR BP MEASURE" };
        displayMessage(skipMsg, 2);

        programState = AWAITING_BP_START;
      }
      break;

    case AWAITING_BP_START:
      // Đợi nhấn nút để bắt đầu đo huyết áp
      if (digitalRead(BUTTON_PIN) == LOW) {
        while (digitalRead(BUTTON_PIN) == LOW) {}  // Chờ thả nút

        // Tắt MAX30100 trước khi đo huyết áp để tránh xung đột I2C
        pox.shutdown();

        state = INCREASING;
        dataCount = 0;
        Serial.println("Starting blood pressure measurement...");
        Serial.println("State: INCREASING");

        // Hiển thị thông báo đang đo huyết áp
        const char* bpMsg[] = { "MEASURING", "BLOOD PRESSURE" };
        displayMessage(bpMsg, 2);

        programState = MEASURING_BP;
      }
      break;

    case MEASURING_BP:
      // Xử lý nút nhấn để dừng khẩn cấp
      if (digitalRead(BUTTON_PIN) == LOW && state != IDLE) {
        while (digitalRead(BUTTON_PIN) == LOW) {}
        state = IDLE;
        Serial.println("State: IDLE in STOP EMERGENCY");

        const char* stopMsg[] = { "MEASUREMENT", "STOPPED" };
        displayMessage(stopMsg, 2);

        programState = AWAITING_BP_START;
        check_state();

        delay(1000);  

        const char* readyMsg[] = { "PRESS BUTTON", "TO RESTART BP" };
        displayMessage(readyMsg, 2);

        dataCount = 0;
      }

      // Hiển thị áp suất hiện tại trên màn hình
      if (currentMillis - tsLastReport > 500) {  // Cập nhật mỗi 500ms
        tsLastReport = currentMillis;

        char pressureStr[20];
        sprintf(pressureStr, "%.0f mmHg", pressure);

        M5.Display.fillScreen(BLACK);
        M5.Display.setTextSize(1.5);
        M5.Display.drawString("MEASURING BP", 64, 40);
        M5.Display.drawString(pressureStr, 64, 80);
      }

      // Lấy mẫu dữ liệu theo khoảng thời gian
      if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentMillis;

        // Lưu dữ liệu áp suất vào mảng
        if (state == RELEASING && dataCount < MAX_SAMPLES) {
          dataArr[dataCount++] = pressure;
        }

        check_state();

        // Kiểm tra khi quá trình thu thập dữ liệu kết thúc
        if (state == IDLE && dataCount > 0) {
          check_state();

          Serial.println("Data collection completed. Publishing data to MQTT...");

          // Publish BP data to MQTT for Python processing
          for (int i = 0; i < dataCount; i++) {
            bool result = mqttClient.publish(mqtt_topic, String(dataArr[i]).c_str());

            if (i % 10 == 9) {
              delay(100); // Small delay every 10 messages to avoid overloading the broker
            }
          }

          Serial.println();
          Serial.print("Total samples published: ");
          Serial.println(dataCount);
          
          // Display waiting message
          M5.Display.fillScreen(BLACK);
          M5.Display.setTextSize(2);
          M5.Display.drawString("PROCESSING", 64, 40);
          M5.Display.drawString("PLEASE WAIT", 64, 70);
          
          // Wait for BP results to come back via MQTT
          // The processing will continue in the MQTT callback
          
          dataCount = 0;
        }
      }
      break;

    case COMPLETED:
      // Display results only once when entering this state
      if (bp_data_received) {
        bp_data_received = false; // Reset for next time
        
        // Hiển thị kết quả đo trên màn hình
        M5.Display.fillScreen(BLACK);
        M5.Display.setTextSize(2);

        char hrStr[20], spo2Str[20], sbpStr[20], dbpStr[20], riskStr[30];
        sprintf(hrStr, "HR: %.1f", finalHeartRate);
        sprintf(spo2Str, "SpO2: %d%%", finalSpO2);
        sprintf(sbpStr, "SBP: %.0f", SBP_value);
        sprintf(dbpStr, "DBP: %.0f", DBP_value);
        sprintf(riskStr, "Risk: %s", riskResult.c_str());

        M5.Display.drawString("RESULTS:", 64, 20);
        M5.Display.setTextSize(1);
        M5.Display.drawString(hrStr, 64, 45);
        M5.Display.drawString(spo2Str, 64, 65);
        M5.Display.drawString(sbpStr, 64, 85);
        M5.Display.drawString(dbpStr, 64, 105);

        // Đặt màu cho kết quả nguy cơ
        if (riskResult == "High Risk") {
          M5.Display.setTextColor(RED);
        } else {
          M5.Display.setTextColor(GREEN);
        }
        M5.Display.drawString(riskStr, 64, 125);
        M5.Display.setTextColor(WHITE);
        M5.Display.drawString("PRESS BUTTON TO RESTART", 64, 145);
      }

      // Đợi nhấn nút để bắt đầu một phiên đo mới
      if (digitalRead(BUTTON_PIN) == LOW) {
        while (digitalRead(BUTTON_PIN) == LOW) {}

        // Khởi động lại MAX30100
        pox.resume();

        // Reset các biến
        fingerDetected = false;
        spo2Ready = false;
        stableCount = 0;
        measurementCount = 0;
        state = IDLE;
        bp_data_received = false;
        SBP_value = 0;
        DBP_value = 0;
        programState = WAITING_FOR_FINGER;

        // Hiển thị thông báo ban đầu
        const char* startMsg[] = { "PUT FINGER", "ON SENSOR" };
        displayMessage(startMsg, 2);

        Serial.println("Starting new measurement session");
        Serial.println("Place finger on sensor to measure SpO2 and Heart Rate");
      }
      break;
  }
}

