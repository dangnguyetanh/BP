import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage.restoration import denoise_wavelet

# === Load CSV ===
def load_csv(filename):
    pressures = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                pressures.append(float(row[1]))
            except:
                continue
    return np.array(pressures)

# === SOS filter without scipy ===
def sos_filter(data, sos):
    w = [[0.0, 0.0] for _ in range(len(sos))]  # state per stage
    y_out = []
    for x in data:
        for i, s in enumerate(sos):
            b0, b1, b2, a0, a1, a2 = s
            if a0 != 1.0:
                b0 /= a0
                b1 /= a0
                b2 /= a0
                a1 /= a0
                a2 /= a0
            w0 = x - a1 * w[i][0] - a2 * w[i][1]
            y = b0 * w0 + b1 * w[i][0] + b2 * w[i][1]
            w[i][1] = w[i][0]
            w[i][0] = w0
            x = y
        y_out.append(y)
    return np.array(y_out)

# === Envelope tools ===
def hl_extrema_idx(s, mode="min"):
    return np.array([i for i in range(1, len(s)-1)
                     if (mode == "min" and s[i] < s[i-1] and s[i] < s[i+1]) or
                        (mode == "max" and s[i] > s[i-1] and s[i] > s[i+1])])

def linear_interpolate(x, y, xi):
    yi = np.zeros(len(xi))
    for i in range(len(xi)):
        if xi[i] <= x[0]: yi[i] = y[0]
        elif xi[i] >= x[-1]: yi[i] = y[-1]
        else:
            for j in range(len(x)-1):
                if x[j] <= xi[i] <= x[j+1]:
                    t = (xi[i] - x[j]) / (x[j+1] - x[j])
                    yi[i] = y[j] + t * (y[j+1] - y[j])
                    break
    return yi

# === Main logic ===
def plot_full_map_analysis(csv_file):
    original = load_csv(csv_file)
    
    # Apply SOS filter first
    sos = [
        [0.23411641, 0.46823283, 0.23411641, 1.0, -0.00244205, 0.27115388],
        [1.0, -2.0, 1.0, 1.0, -1.57146442, 0.6722172]
    ]
    y_raw = sos_filter(original, sos)
    filtered = np.array([
        y * 45 if (i > 50 and abs(y) < 4) else 0
        for i, y in enumerate(y_raw)
    ])
    
    # Apply wavelet denoising with adjusted parameters
    denoised_signal = denoise_wavelet(filtered, method='BayesShrink', mode='soft', wavelet='sym8', wavelet_levels=3, sigma=3.5, rescale_sigma=False)
    
    # Optional secondary light denoising to target residual noise
    final_denoised = denoise_wavelet(denoised_signal, method='BayesShrink', mode='soft', wavelet='sym8', wavelet_levels=4, sigma=1.0, rescale_sigma=False)
    
    x = np.arange(len(final_denoised))

    lmin = hl_extrema_idx(final_denoised, mode="min")
    lower_curve = linear_interpolate(lmin, final_denoised[lmin], x)
    amplitude = final_denoised - lower_curve
    lmax = hl_extrema_idx(amplitude, mode="max")
    envelope = linear_interpolate(lmax, amplitude[lmax], x)

    MAP_amp = np.max(envelope)
    MAP_idx = np.argmax(envelope)
    
    SBP_amp = 0.487 * MAP_amp
    DBP_amp = 0.658 * MAP_amp

    SBP_idx = next(i for i in range(MAP_idx-1, -1, -1) if envelope[i] <= SBP_amp)
    DBP_idx = next(i for i in range(MAP_idx+1, len(envelope)) if envelope[i] <= DBP_amp)
    
    print(f"MAP = {original[MAP_idx]}")
    print(f"SBP = {original[SBP_idx]}")
    print(f"DBP = {original[DBP_idx]}")

    plt.figure(figsize=(12, 6))
    plt.plot(x, original, label="Original Data", color='blue', alpha=0.5)
    plt.plot(x, final_denoised, label="Final Denoised Data", color='green')
    plt.plot(x, amplitude, label="BP Amplitude", color='orange')
    plt.plot(x, envelope, label="Amplitude Envelope", color='mediumorchid')
    plt.axhline(y=MAP_amp, color='black', linestyle='--', label="MAP")
    plt.axhline(y=DBP_amp, color='purple', linestyle='--', label="DIA")
    plt.axhline(y=SBP_amp, color='blue', linestyle='--', label="SYS")
    plt.scatter([MAP_idx], [original[MAP_idx]], color='black', label=f"MAP = {original[MAP_idx]:.2f}", zorder=5)
    plt.scatter([SBP_idx], [original[SBP_idx]], color='orange', label=f"SYS = {original[SBP_idx]:.2f}", zorder=5)
    plt.scatter([DBP_idx], [original[DBP_idx]], color='orange', label=f"DIA = {original[DBP_idx]:.2f}", zorder=5)
    plt.title("Original vs Denoised vs Filtered Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_full_map_analysis("original_data.csv")