#include <Arduino.h>
#include <Wire.h> 
#include "MAX30100_PulseOximeter.h"
#include <M5AtomS3.h>

#define REPORTING_PERIOD_MS 1000
#define NUM_MEASUREMENTS 10
#define MIN_VALID_MEASUREMENTS 5
#define HR_ERROR_MARGIN 5.0   // Sai số cho phép cho nhịp tim (bpm)
#define SPO2_ERROR_MARGIN 2.0 // Sai số cho phép cho SpO2 (%)

PulseOximeter pox;
uint32_t tsLastReport = 0;
uint32_t measurementStartTime = 0;

// Mảng để lưu các kết quả đo
float heartRates[NUM_MEASUREMENTS];
int spo2Values[NUM_MEASUREMENTS];
int measurementCount = 0;
bool measurementInProgress = false;
bool measurementComplete = false;
bool fingerDetected = false;  // Add flag for finger detection

// Callback khi phát hiện nhịp tim
void onBeatDetected() {
  Serial.println("💓 Beat Detected!");
  fingerDetected = true;  // Set flag when beat is detected
}

// Hàm hiển thị thông báo lên màn hình
void displayMessage(const char* messages[], int lines, int textSize = 1, int x = 64, int y = 64) {
  M5.Display.fillScreen(BLACK);
  M5.Display.setTextColor(WHITE);
  M5.Display.setTextSize(textSize);
  M5.Display.setTextDatum(middle_center);
  int lineHeight = textSize * 10;
  for (int i = 0; i < lines; i++) {
    int yOffset = y + (i - lines / 2) * lineHeight;
    M5.Display.drawString(messages[i], x, yOffset);
  }
}

// Hàm kiểm tra xem có ít nhất MIN_VALID_MEASUREMENTS kết quả tương đương nhau không
bool analyzeResults(float& finalHR, int& finalSpO2) {
  int maxHRMatches = 0;
  int maxSpO2Matches = 0;
  float bestHR = 0;
  int bestSpO2 = 0;
  
  // Kiểm tra từng kết quả
  for (int i = 0; i < NUM_MEASUREMENTS; i++) {
    // Bỏ qua kết quả không hợp lệ (0 hoặc giá trị quá thấp)
    if (heartRates[i] < 40 || spo2Values[i] < 70) continue;
    
    int hrMatches = 0;
    int spo2Matches = 0;
    
    // So sánh với tất cả các kết quả khác
    for (int j = 0; j < NUM_MEASUREMENTS; j++) {
      if (abs(heartRates[i] - heartRates[j]) <= HR_ERROR_MARGIN) {
        hrMatches++;
      }
      if (abs(spo2Values[i] - spo2Values[j]) <= SPO2_ERROR_MARGIN) {
        spo2Matches++;
      }
    }
    
    // Cập nhật giá trị tốt nhất nếu có nhiều kết quả tương đồng hơn
    if (hrMatches > maxHRMatches) {
      maxHRMatches = hrMatches;
      bestHR = heartRates[i];
    }
    
    if (spo2Matches > maxSpO2Matches) {
      maxSpO2Matches = spo2Matches;
      bestSpO2 = spo2Values[i];
    }
  }
  
  // Lưu giá trị cuối cùng
  finalHR = bestHR;
  finalSpO2 = bestSpO2;
  
  // Kiểm tra xem có đủ số lượng kết quả tương đồng không
  return (maxHRMatches >= MIN_VALID_MEASUREMENTS && maxSpO2Matches >= MIN_VALID_MEASUREMENTS);
}

// Hàm bắt đầu quá trình đo mới
void startNewMeasurement() {
  measurementInProgress = true;
  measurementComplete = false;
  measurementCount = 0;
  fingerDetected = false;  // Reset finger detection flag
  
  // Hiển thị hướng dẫn
  const char* instructions[] = {"PUT FINGER", "ON SENSOR"};
  displayMessage(instructions, 2, 1, 64, 64);
  
  // Reset mảng kết quả
  for (int i = 0; i < NUM_MEASUREMENTS; i++) {
    heartRates[i] = 0;
    spo2Values[i] = 0;
  }
}

void setup() {
  M5.begin();
  Serial.begin(115200);
  Wire.begin(2, 1);
  
  if (!pox.begin()) {
    Serial.println("❌ FAILED to initialize MAX30100");
    const char* errorMsg[] = {"ERROR SENSOR"};
    displayMessage(errorMsg, 1, 2, 64, 64);
    while (1) delay(100);
  }
  
  pox.setIRLedCurrent(MAX30100_LED_CURR_17_4MA);
  pox.setOnBeatDetectedCallback(onBeatDetected);

  Serial.println("✅ MAX30100 initialized!");
  
  // Bắt đầu quy trình đo
  startNewMeasurement();
}

void loop() {
  M5.update();
  pox.update();
  
  // Nếu quá trình đo đã hoàn thành, kiểm tra nút để bắt đầu đo lại
  if (measurementComplete) {
    if (M5.BtnA.wasPressed()) {
      startNewMeasurement();
    }
    return;
  }
  
  // Nếu đang trong quá trình đo
  if (measurementInProgress) {
    if (millis() - tsLastReport > REPORTING_PERIOD_MS) {
      float hr = pox.getHeartRate();
      int spo2 = pox.getSpO2();
      
      Serial.print("Heart Rate : ");
      Serial.print(hr);
      Serial.print(" bpm  |  SpO2 : ");
      Serial.print(spo2);
      Serial.println(" %");
      
      // Hiển thị thông tin dựa trên trạng thái phát hiện ngón tay
      if (!fingerDetected) {
        const char* instructions[] = {"PUT FINGER", "ON SENSOR"};
        displayMessage(instructions, 2, 1, 64, 64);
      } else {
        const char* status[] = {"MEASURING..."};
        displayMessage(status, 1, 1, 64, 64);
      }
      
      // Lưu kết quả hợp lệ (loại bỏ các giá trị 0 hoặc quá thấp)
      if (hr > 40 && spo2 > 70) {
        heartRates[measurementCount] = hr;
        spo2Values[measurementCount] = spo2;
        measurementCount++;
      }
      
      tsLastReport = millis();
      
      // Kiểm tra xem đã đủ số lần đo chưa
      if (measurementCount >= NUM_MEASUREMENTS) {
        measurementInProgress = false;
        measurementComplete = true;
        
        // Phân tích kết quả
        float finalHR;
        int finalSpO2;
        bool success = analyzeResults(finalHR, finalSpO2);
        
        if (success) {
          // Hiển thị kết quả thành công
          char resultHR[30], resultSpO2[30];
          sprintf(resultHR, "HR: %.1f bpm", finalHR);
          sprintf(resultSpO2, "SpO2: %d%%", finalSpO2);
          const char* results[] = {resultHR, resultSpO2};
          displayMessage(results, 2, 1, 64, 64);
        } else {
          // Hiển thị thông báo thất bại
          const char* failMsg[] = {"CAN NOT", "MEASURE"};
          displayMessage(failMsg, 2, 1, 64, 64);
        }
      }
    }
  }
}