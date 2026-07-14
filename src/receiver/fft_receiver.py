import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
# ======================
# HackRF settings
# ======================
FREQ = "915000000"
FS = 8_000_000
cmd = [
    "hackrf_transfer",
    "-r", "-",
    "-f", FREQ,
    "-s", str(FS),
    "-g", "20",
    "-l", "32"
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)


# ======================
# Parameters
# ======================

CHUNK = 4_000_000  # 0.5 sec — freq resolution = 2 Hz

DISPLAY_SECONDS = 30
CHUNK_DURATION = CHUNK / FS
points_in_window = int(DISPLAY_SECONDS / CHUNK_DURATION)
time_axis = np.linspace(0, DISPLAY_SECONDS, points_in_window)


# ======================
# Plot
# ======================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

display_buffer = deque([0.0] * points_in_window, maxlen=points_in_window)
line1, = ax1.plot(time_axis, list(display_buffer), 'b-o', linewidth=2, markersize=3)
ax1.set_title("Demodulated BPSK")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Signed Amplitude")
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlim(0, DISPLAY_SECONDS)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.grid()

bit_buffer = deque([0.5] * points_in_window, maxlen=points_in_window)
line2, = ax2.plot(time_axis, list(bit_buffer), 'r-s', linewidth=2, markersize=5)
ax2.set_title("Bit Decisions")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Bit")
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlim(0, DISPLAY_SECONDS)
ax2.set_yticks([0, 1])
ax2.grid()

plt.tight_layout()


# ======================
# State
# ======================

prev_phasor = None
current_bit = 1
# Keep last N diff_angles to smooth decisions
diff_history = deque(maxlen=5)


# ======================
# Update
# ======================

def update(frame):
    global prev_phasor, current_bit

    raw = proc.stdout.read(2 * CHUNK)
    if len(raw) < 2 * CHUNK:
        return line1, line2

    data = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0
    iq = data[0::2] + 1j * data[1::2]
    N = len(iq)

    # Check signal power
    power = np.mean(np.abs(iq) ** 2)

    if power < 0.01:
        print(f"NO SIGNAL (power={power:.4f})")
        display_buffer.append(0.0)
        bit_buffer.append(0.5)
        prev_phasor = None
        line1.set_ydata(list(display_buffer))
        line2.set_ydata(list(bit_buffer))
        return line1, line2

    # -----------------------------------------
    # Windowed FFT on FULL chunk for best freq resolution
    # Hann window reduces spectral leakage
    # -----------------------------------------
    window = np.hanning(N)
    fft_vals = np.fft.fft(iq * window)
    freqs = np.fft.fftfreq(N, d=1.0 / FS)
    mag = np.abs(fft_vals)
    mag[0] = 0
    valid = np.abs(freqs) < 500000
    mag[~valid] = 0
    peak_idx = np.argmax(mag)

    # Parabolic interpolation for sub-Hz accuracy
    if 1 < peak_idx < N - 1:
        a = mag[peak_idx - 1]
        b = mag[peak_idx]
        c = mag[peak_idx + 1]
        denom = (a - 2*b + c)
        if denom != 0:
            delta = 0.5 * (a - c) / denom
        else:
            delta = 0
        freq_hz = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
    else:
        freq_hz = freqs[peak_idx]

    # -----------------------------------------
    # Mix down and get phasor
    # -----------------------------------------
    n = np.arange(N)
    mixer = np.exp(-1j * 2 * np.pi * freq_hz / FS * n)
    baseband = iq * mixer
    chunk_phasor = np.mean(baseband)
    amplitude = np.abs(chunk_phasor)

    # -----------------------------------------
    # Differential: compare to previous phasor
    # -----------------------------------------
    if prev_phasor is not None:
        diff = chunk_phasor * np.conj(prev_phasor)
        diff_angle = np.degrees(np.angle(diff))

        # Track angle to detect clear transitions
        diff_history.append(diff_angle)

        # Only flip bit on clear ~180° change
        if abs(diff_angle) > 120:
            current_bit = 1 - current_bit

        # Squared amplitude with sign
        sign = 1.0 if current_bit == 1 else -1.0
        demod_val = sign * amplitude

        print(f"BIT = {current_bit}  amp={amplitude:.4f}  freq={freq_hz:.0f} Hz  diff={diff_angle:.1f}°")
    else:
        demod_val = amplitude
        current_bit = 1
        print(f"FIRST  amp={amplitude:.4f}  freq={freq_hz:.0f} Hz")

    prev_phasor = chunk_phasor

    display_buffer.append(demod_val)
    bit_buffer.append(float(current_bit))

    line1.set_ydata(list(display_buffer))
    line2.set_ydata(list(bit_buffer))
    return line1, line2


ani = FuncAnimation(fig, update, interval=100)
plt.show()
proc.kill()
