#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // Thay micro_log.h
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>

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

void setup() {
  Serial.begin(115200);

  // Thiết lập ErrorReporter
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
}

void loop() {
  // Thay bằng dữ liệu từ cảm biến
  float raw_data[4] = {60.0, 95.7, 124.0, 86.0}; // Heart Rate,Oxygen Saturation,Systolic Blood Pressure,Diastolic Blood Pressure

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
    return;
  }

  // Lấy kết quả
  float probability = output->data.f[0];
  String risk = (probability > 0.5) ? "High Risk" : "Low Risk";
  Serial.print("Probability: ");
  Serial.print(probability);
  Serial.print(" | Risk: ");
  Serial.println(risk);

  delay(1000);
}