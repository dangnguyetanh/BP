#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // Thay micro_log.h
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>

namespace {
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroAllocator* allocator = nullptr;
} 

float scaler_mean[4] = {79.53374663,97.50437243, 124.4379712, 79.49962504  };
float scaler_scale[4] = { 11.55286498, 1.44259433,8.65692398, 5.75723374 };
