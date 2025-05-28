#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

#include <M5AtomS3.h>


#define SLOW_RELEASE_PIN      6
#define PUMP_PIN0             7
#define PUMP_PIN1             8
#define TURN_OFF_PIN0         38
#define TURN_OFF_PIN1         39
#define BUTTON_PIN            41
void onPump() {
  digitalWrite(PUMP_PIN0, HIGH);
  digitalWrite(PUMP_PIN1, LOW);
}
void offPump() {
  digitalWrite(PUMP_PIN0, LOW);
  digitalWrite(PUMP_PIN1, LOW);
}
void onSlowRelease() {
  digitalWrite(SLOW_RELEASE_PIN, LOW);
}
void offSlowRelease() {
  digitalWrite(SLOW_RELEASE_PIN, HIGH);
}
void onTurnOff() {
  digitalWrite(TURN_OFF_PIN0, HIGH);
  digitalWrite(TURN_OFF_PIN1, LOW);
}
void offTurnOff() {
  digitalWrite(TURN_OFF_PIN0, LOW);
  digitalWrite(TURN_OFF_PIN1, LOW);
}
Adafruit_ADS1115 ads;

// Hằng số
const double slope     = 64.64F;
const double intercept = -32.32F;
const double PRESSURE_MAX     = 150.0F;
const double PRESSURE_RELEASE =  50.0F;

// State machine
enum State { IDLE, INCREASING, RELEASING };
State state = IDLE;

// Mảng lưu dữ liệu áp suất
const int MAX_SAMPLES = 1000;
double dataArr[MAX_SAMPLES];
int  dataCount = 0;

// Thời gian mẫu
const unsigned long SAMPLE_INTERVAL = 100;
unsigned long lastSampleTime = 0;
unsigned long lastPrintTime = 0;

double pressure = 0.0F;
void read_sensor() {
  unsigned long now = millis();
    int16_t raw = ads.readADC_SingleEnded(2);
    double voltage = raw * 0.0001875F;
    pressure = slope * voltage + intercept;

    if (now - lastPrintTime >= 1000) {
      lastPrintTime = now;
      Serial.println(pressure, 1);
    }
}

void check_state() {
  read_sensor();
  switch (state) {
    case IDLE:
      // Ở trạng thái IDLE: Tắt bơm và bật cả hai van xả
      offPump();
      onSlowRelease();  // Bật van xả chậm
      onTurnOff();      // Bật van xả thứ hai
      break;
      
    case INCREASING:
      // Ở trạng thái INCREASING: Bật bơm, tắt cả hai van xả
      onPump();
      offSlowRelease();  // Tắt van xả chậm
      offTurnOff();      // Tắt van xả thứ hai
      
      // Khi áp suất đạt mức tối đa, chuyển sang trạng thái RELEASING
      if (pressure >= PRESSURE_MAX) {
        Serial.println("→ Releasing phase start");
        state = RELEASING;
      }
      break;
      
    case RELEASING:
      // Ở trạng thái RELEASING: Tắt bơm, bật van xả chậm, tắt van xả thứ hai
      offPump();
      onSlowRelease();  // Bật van xả chậm
      offTurnOff();     // Tắt van xả thứ hai để xả chậm
      
      // Khi áp suất xuống dưới mức giới hạn, chuyển sang trạng thái IDLE
      if (pressure <= PRESSURE_RELEASE) {
        Serial.println("→ Releasing phase end");
        state = IDLE;
        // Khi chuyển sang IDLE, cả hai van xả đều được bật trong case IDLE
      }
      break;

    default:
      break;
  }
}