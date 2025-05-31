#include <Arduino.h>
#include <Wire.h>
#include <M5AtomS3.h>
#include "process.h"
#include "bp.h"
#include "MAX30100_PulseOximeter.h"
#include "mqtt.h"
#include <ArduinoJson.h>

#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>

// Cấu hình mô hình AI
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

// Thông số chuẩn hóa
float scaler_mean[4] = { 79.53374663, 97.50437243, 124.4379712, 79.49962504 };
float scaler_scale[4] = { 11.55286498, 1.44259433, 8.65692398, 5.75723374 };

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
String riskResult = "";          // Kết quả đánh giá nguy cơ

// Enum trạng thái chương trình
enum ProgramState {
  WAITING_FOR_FINGER,
  MEASURING_HR_SPO2,
  AWAITING_BP_START,
  MEASURING_BP,
  COMPLETED
};

ProgramState programState = WAITING_FOR_FINGER;

// Callback được gọi khi phát hiện nhịp tim
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

// Hàm đánh giá nguy cơ sức khỏe sử dụng mô hình AI
String assessHealthRisk(float hr, int spo2, float sbp, float dbp) {
  // Tạo mảng dữ liệu cho mô hình
  float raw_data[4] = { hr, (float)spo2, sbp, dbp };

  Serial.println("Measurements array:");
  for (int i = 0; i < 4; i++) {
    Serial.print(raw_data[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Chuẩn hóa dữ liệu
  float input_data[4];
  for (int i = 0; i < 4; i++) {
    input_data[i] = (raw_data[i] - scaler_mean[i]) / scaler_scale[i];
  }

  // Đưa dữ liệu vào mô hình
  for (int i = 0; i < 4; i++) {
    input->data.f[i] = input_data[i];
  }

  // Chạy suy luận
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return "Error";
  }

  // Lấy kết quả
  float probability = output->data.f[0];
  String risk = (probability > 0.5) ? "High Risk" : "Low Risk";
  Serial.print("Probability: ");
  Serial.print(probability);
  Serial.print(" | Risk: ");
  Serial.println(risk);

  return risk;
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
  M5.begin();
  Serial.begin(115200);
  Wire.begin(2, 1);

  connectWiFi();
  mqttClient.setServer(mqtt_server, mqtt_port);
  mqttReconnect();
  mqttClient.publish(mqtt_topic, "hello from ESP32");
  // Configure NTP for real-time
  configTime(7 * 3600, 0, "pool.ntp.org");  // UTC+7 for Vietnam
  while (!time(nullptr)) {
    Serial.println("Waiting for NTP time sync...");
    delay(1000);
  }
  Serial.println("Time synchronized");

  // Thiết lập ErrorReporter cho AI
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load mô hình AI
  model = tflite::GetModel(health_risk_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version mismatch!");
    const char* modelErrorMsg[] = { "MODEL ERROR", "VERSION MISMATCH" };
    displayMessage(modelErrorMsg, 2);
    while (1) delay(100);
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
    const char* tensorErrorMsg[] = { "AI ERROR", "TENSOR ALLOCATION" };
    displayMessage(tensorErrorMsg, 2);
    while (1) delay(100);
  }

  // Lấy tensor input và output
  input = interpreter->input(0);
  output = interpreter->output(0);

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
        sprintf(hrStr, "HR: %.1f bpm", heartRate);
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
            M5.Display.drawString("PRESS BUTTON FOR BP MEASURE", 64, 120);

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

        // Thay đổi ở đây - chuyển sang trạng thái AWAITING_BP_START thay vì COMPLETED
        programState = AWAITING_BP_START;
        check_state();

        // Hiển thị thông báo sẵn sàng đo lại
        delay(2000);  // Chờ 2 giây để người dùng đọc thông báo "MEASUREMENT STOPPED"

        const char* readyMsg[] = { "PRESS BUTTON", "TO RESTART BP" };
        displayMessage(readyMsg, 2);

        // Reset biến đếm dữ liệu
        dataCount = 0;
      }

      // Hiển thị áp suất hiện tại trên màn hình
      if (currentMillis - tsLastReport > 500) {  // Cập nhật mỗi 500ms
        tsLastReport = currentMillis;

        char pressureStr[20];
        sprintf(pressureStr, "%.0f mmHg", pressure);

        M5.Display.fillScreen(BLACK);
        M5.Display.setTextSize(2);
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
          // bp_process(dataArr, dataCount);

          // Đánh giá nguy cơ sức khỏe sử dụng AI
          riskResult = assessHealthRisk(finalHeartRate, finalSpO2, SBP_value, DBP_value);

          Serial.println("Data collection completed. Printing data:");

          // In dữ liệu áp suất
          for (int i = 0; i < dataCount; i++) {
            Serial.print(dataArr[i]);
            Serial.print(", ");
            if (i % 10 == 9) {
              delay(100);
            }
          }

          Serial.println();
          Serial.print("Total samples collected: ");
          Serial.println(dataCount);


StaticJsonDocument<256> doc;  // Increased size to accommodate risk field
doc["systolic"] = round(SBP_value);         // Làm tròn thành số nguyên
doc["diastolic"] = round(DBP_value);        // Làm tròn thành số nguyên
doc["heart_rate"] = round(finalHeartRate);  // Làm tròn thành số nguyên
doc["spo2"] = finalSpO2;                    // SpO2 đã là số nguyên
doc["risk"] = riskResult;                   // Thêm kết quả đánh giá nguy cơ
doc["timestamp"] = getTimestamp();          // Thời gian

// Serialize JSON thành chuỗi
char jsonBuffer[256];  // Increased buffer size
serializeJson(doc, jsonBuffer);

// Gửi lên MQTT broker
if (mqttClient.publish(mqtt_topic, jsonBuffer)) {
  Serial.println("Published to MQTT: ");
  Serial.println(jsonBuffer);
} else {
  Serial.println("Failed to publish to MQTT");
}
          // In kết quả đo
          Serial.print("SpO2: ");
          Serial.print(finalSpO2);
          Serial.print("%, Heart Rate: ");
          Serial.print(finalHeartRate);
          Serial.print(" bpm, SBP: ");
          Serial.print(SBP_value);
          Serial.print(" mmHg, DBP: ");
          Serial.print(DBP_value);
          Serial.print(" mmHg, Risk: ");
          Serial.println(riskResult);

          // Hiển thị kết quả đo trên màn hình
          M5.Display.fillScreen(BLACK);
          M5.Display.setTextSize(2);

          char hrStr[20], spo2Str[20], sbpStr[20], dbpStr[20], riskStr[20];
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

          dataCount = 0;
          programState = COMPLETED;
          Serial.println("\nMeasurement session finished.");
        }
      }
      break;

    case COMPLETED:
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