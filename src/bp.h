double SBP_value ;
double DBP_value;
void sos_filter(double* data, int data_length, double* output, double sos[][6], int sos_stages) {
  // Mảng trạng thái cho mỗi stage của bộ lọc (lưu trữ giá trị của w[0] và w[1])
  double w[sos_stages][2] = { { 0.0, 0.0 } };  // Khởi tạo trạng thái ban đầu

  // Duyệt qua từng mẫu dữ liệu
  for (int n = 0; n < data_length; n++) {
    double x = data[n];  // Lấy mẫu dữ liệu hiện tại

    // Áp dụng bộ lọc SOS
    for (int i = 0; i < sos_stages; i++) {
      // Lấy các hệ số của stage i
      double b0 = sos[i][0];
      double b1 = sos[i][1];
      double b2 = sos[i][2];
      double a0 = sos[i][3];
      double a1 = sos[i][4];
      double a2 = sos[i][5];

      // Đảm bảo a0 = 1
      if (a0 != 1.0) {
        b0 /= a0;
        b1 /= a0;
        b2 /= a0;
        a1 /= a0;
        a2 /= a0;
      }

      // Tính toán giá trị w0 (biến trung gian)
      double w0 = x - a1 * w[i][0] - a2 * w[i][1];
      // Tính toán giá trị đầu ra y của stage này
      double y = b0 * w0 + b1 * w[i][0] + b2 * w[i][1];

      // Cập nhật trạng thái cho stage tiếp theo
      w[i][1] = w[i][0];
      w[i][0] = w0;

      // Cập nhật giá trị mẫu x cho stage tiếp theo
      x = y;
    }

    // Lưu giá trị đầu ra vào mảng output
    output[n] = x;
  }
}

// Hàm tìm chỉ số của các cực trị (min hoặc max)
void hl_extrema_idx(double* s, int s_length, String mode, int* extrema_idx, int& count) {
  count = 0;  // Khởi tạo số lượng điểm cực trị

  // Duyệt qua từng phần tử trong mảng và tìm các điểm cực trị
  for (int i = 1; i < s_length - 1; i++) {
    if (mode == "min" && s[i] < s[i - 1] && s[i] < s[i + 1]) {
      extrema_idx[count++] = i;  // Thêm chỉ số min vào mảng kết quả
    } else if (mode == "max" && s[i] > s[i - 1] && s[i] > s[i + 1]) {
      extrema_idx[count++] = i;  // Thêm chỉ số max vào mảng kết quả
    }
  }
}

// Hàm nội suy tuyến tính
void linear_interpolate(int* x, double* y, int* xi, double* yi, int len_x, int len_xi) {
  for (int i = 0; i < len_xi; i++) {
    double xi_val = (double)xi[i];
    if (xi_val <= (double)x[0]) {
      yi[i] = y[0];
    } else if (xi_val >= (double)x[len_x - 1]) {
      yi[i] = y[len_x - 1];
    } else {
      for (int j = 0; j < len_x - 1; j++) {
        double x_j = (double)x[j];
        double x_j1 = (double)x[j + 1];
        if (x_j <= xi_val && xi_val <= x_j1) {
          double t = (xi_val - x_j) / (x_j1 - x_j);
          yi[i] = y[j] + t * (y[j + 1] - y[j]);
          break;
        }
      }
    }
  }
}

void bp_process(double* test_data, int test_len) {
  double sos[2][6] = {
    { 0.23411641, 0.46823283, 0.23411641, 1.0, -0.00244205, 0.27115388 },
    { 1.0, -2.0, 1.0, 1.0, -1.57146442, 0.6722172 }
  };

  //========================y_raw=================//
  Serial.print("Free heap memory before allocation: ");
  Serial.println(ESP.getFreeHeap());

  double* y_raw = (double*)malloc(test_len * sizeof(double));  // Cấp phát bộ nhớ cho y_raw
  if (y_raw == nullptr) {
    Serial.println("Memory allocation failed for y_raw");
    return;
  }
  sos_filter(test_data, test_len, y_raw, sos, 2);
Serial.print("Free heap memory after y_raw allocation: ");
Serial.println(ESP.getFreeHeap());
  //========================filtered=================//
  // Mảng lưu kết quả filtered
  double* filtered = (double*)malloc(test_len * sizeof(double));  // Cấp phát bộ nhớ cho filtered
  if (filtered == nullptr) {
    Serial.println("Memory allocation failed for filtered");
    free(y_raw);  // Giải phóng bộ nhớ trước đó
    return;
  }
  for (int i = 0; i < test_len; i++) {
    if (i > 50 && abs(y_raw[i]) < 4) {
      filtered[i] = y_raw[i] * 45;  // Nếu điều kiện thỏa, nhân với 45
    } else {
      filtered[i] = 0;  // Nếu không thỏa, gán 0
    }
  }

  //========================x=================//
  int* x = (int*)malloc(test_len * sizeof(int));
  for (int i = 0; i < test_len; i++) {
    x[i] = i;  // Gán giá trị từ 0 đến test_len-1 vào mảng x
  }

  //========================lmin=================//
  int* lmin = (int*)malloc(test_len * sizeof(int));  // Cấp phát bộ nhớ cho lmin
  if (lmin == nullptr) {
    Serial.println("Memory allocation failed for lmin");
    free(y_raw);     // Giải phóng bộ nhớ trước đó
    free(filtered);  // Giải phóng bộ nhớ trước đó
    return;
  }
  int count_min = 0;
  hl_extrema_idx(filtered, test_len, "min", lmin, count_min);

  //========================lower_curve=================//
  double* lower_curve = (double*)malloc(test_len * sizeof(double));  // Cấp phát bộ nhớ cho lower_curve
  if (lower_curve == nullptr) {
    Serial.println("Memory allocation failed for lower_curve");
    return;
  }

  double* y = (double*)malloc(count_min * sizeof(double));
  for (int i = 0; i < count_min; i++) {
    y[i] = filtered[lmin[i]];
  }
  if (count_min < 2) {
    Serial.println("Not enough minima for interpolation");
    return;
  }
  linear_interpolate(lmin, y, x, lower_curve, count_min, test_len);

  //=========================amplitude=================//
  // Cấp phát bộ nhớ cho amplitude
  double* amplitude = (double*)malloc(test_len * sizeof(double));
  if (amplitude == nullptr) {
    Serial.println("Memory allocation failed for amplitude");
    // Giải phóng các mảng khác nếu cần
    return;
  }

  for (int i = 0; i < test_len; i++) {
    amplitude[i] = filtered[i] - lower_curve[i];
  }

  //========================lmax=================//
  int* lmax = (int*)malloc(test_len * sizeof(int));  // Cấp phát bộ nhớ cho lmin
  if (lmax == nullptr) {
    Serial.println("Memory allocation failed for lmax");
    return;
  }
  int count_max = 0;
  hl_extrema_idx(filtered, test_len, "max", lmax, count_max);

  //========================envelop=================//
  double* envelope = (double*)malloc(test_len * sizeof(double));
  if (envelope == nullptr) {
    Serial.println("Memory allocation failed for envelope");
    return;
  }

  // Tạo mảng y_max chứa amplitude tại lmax
  double* y_max = (double*)malloc(count_max * sizeof(double));
  if (y_max == nullptr) {
    Serial.println("Memory allocation failed for y_max");
    return;
  }
  for (int i = 0; i < count_max; i++) {
    y_max[i] = amplitude[lmax[i]];
  }

  // Tính envelope
  linear_interpolate(lmax, y_max, x, envelope, count_max, test_len);


  //============================SBP_amp & DBP_amp===========================//
  double MAP_amp = envelope[0];  // Khởi tạo với giá trị đầu tiên
  int MAP_idx = 0;               // Khởi tạo chỉ số

  for (int i = 1; i < test_len; i++) {
    if (envelope[i] > MAP_amp) {
      MAP_amp = envelope[i];  // Cập nhật giá trị lớn nhất
      MAP_idx = i;            // Cập nhật chỉ số
    }
  }

  double SBP_amp = 0.487 * MAP_amp;
  double DBP_amp = 0.658 * MAP_amp;

  //=========================SBP_idx=================//
  int SBP_idx = -1;
  for (int i = MAP_idx - 1; i >= 0; i--) {
    if (envelope[i] <= SBP_amp) {
      SBP_idx = i;
      break;
    }
  }

  //=========================DBP_idx=================//
  int DBP_idx = -1;
  for (int i = MAP_idx + 1; i < test_len; i++) {
    if (envelope[i] <= DBP_amp) {
      DBP_idx = i;
      break;
    }
  }

  //=========================MAP_value, SBP_value, DBP_value=================//
  double MAP_value = test_data[MAP_idx];
  SBP_value = test_data[SBP_idx];
  DBP_value = test_data[DBP_idx];

  Serial.print("MAP = ");
  Serial.println(MAP_value, 2);  
  Serial.print("SYS = ");
  Serial.println(SBP_value, 2);
  Serial.print("DIA = ");
  Serial.println(DBP_value, 2);

  free(y_raw);
  free(filtered);
  free(lmin);
  free(lower_curve);
  free(y);
  free(amplitude);
  free(lmax);
  free(envelope);
  free(y_max);
  free(x);
}