import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from numba import njit


# ======================
# Numba-accelerated Costas Loop
# ======================

@njit(cache=True)
def costas_loop(I_raw, Q_raw, loop_phase, loop_freq, alpha, beta):
    """
    BPSK Costas loop - runs sample by sample at full speed.
    Returns: demodulated I output, final loop_phase, final loop_freq
    """
    N = len(I_raw)
    out_I = np.empty(N)

    for i in range(N):
        cos_v = np.cos(loop_phase)
        sin_v = np.sin(loop_phase)

        i_out = I_raw[i] * cos_v + Q_raw[i] * sin_v
        q_out = -I_raw[i] * sin_v + Q_raw[i] * cos_v

        out_I[i] = i_out

        if i_out > 0:
            error = q_out
        else:
            error = -q_out

        loop_freq += beta * error
        loop_phase += loop_freq + alpha * error

        if loop_phase > np.pi:
            loop_phase -= 2.0 * np.pi
        elif loop_phase < -np.pi:
            loop_phase += 2.0 * np.pi

    return out_I, loop_phase, loop_freq


# ======================
# HackRF settings
# ======================

FREQ = "915000000"
FS = 2_000_000  # 2 MHz — matches TX

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
# Costas Loop Parameters
# ======================

LOOP_BW = 1000  # Hz — 1 kHz bandwidth
zeta = 0.707
BnT = LOOP_BW / FS
denom = 1 + 2*zeta*BnT + BnT**2
alpha_loop = (4 * zeta * BnT) / denom
beta_loop = (4 * BnT**2) / denom

# Compensate for signal amplitude ~0.5
amp_scale = 2.0
alpha_loop *= amp_scale
beta_loop *= amp_scale

print(f"Loop params: alpha={alpha_loop:.6f}, beta={beta_loop:.9f}, BW={LOOP_BW} Hz")


# ======================
# Parameters
# ======================

# 500k samples per frame = 0.25 sec at 2 MHz
CHUNK = 500_000

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
ax1.set_title("Demodulated BPSK (Costas Loop + Numba)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
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

loop_phase = 0.0
loop_freq = 0.0
freq_acquired = False

# Warm up numba
print("Warming up Numba JIT...")
dummy_I = np.zeros(100, dtype=np.float64)
dummy_Q = np.zeros(100, dtype=np.float64)
_ = costas_loop(dummy_I, dummy_Q, 0.0, 0.0, alpha_loop, beta_loop)
print("Ready!")


# ======================
# Update
# ======================

def update(frame):
    global loop_phase, loop_freq, freq_acquired

    raw = proc.stdout.read(2 * CHUNK)
    if len(raw) < 2 * CHUNK:
        return line1, line2

    data = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0
    I_raw = data[0::2]
    Q_raw = data[1::2]

    # Check signal power
    power = np.mean(I_raw**2 + Q_raw**2)

    if power < 0.01:
        print(f"NO SIGNAL (power={power:.4f})")
        display_buffer.append(0.0)
        bit_buffer.append(0.5)
        line1.set_ydata(list(display_buffer))
        line2.set_ydata(list(bit_buffer))
        return line1, line2

    # -----------------------------------------
    # Acquire frequency using FFT (first frame only)
    # -----------------------------------------
    if not freq_acquired:
        N = len(I_raw)
        iq = I_raw + 1j * Q_raw
        fft_vals = np.fft.fft(iq)
        freqs = np.fft.fftfreq(N, d=1.0 / FS)
        mag = np.abs(fft_vals)
        mag[0] = 0
        valid = np.abs(freqs) < 500000
        mag[~valid] = 0
        peak_idx = np.argmax(mag)

        if 1 < peak_idx < N - 1:
            a = mag[peak_idx - 1]
            b = mag[peak_idx]
            c = mag[peak_idx + 1]
            denom_val = (a - 2*b + c)
            if denom_val != 0:
                delta = 0.5 * (a - c) / denom_val
            else:
                delta = 0
            freq_hz_est = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
        else:
            freq_hz_est = freqs[peak_idx]

        loop_freq = 2.0 * np.pi * freq_hz_est / FS
        freq_acquired = True
        print(f"*** FREQ ACQUIRED: {freq_hz_est:.0f} Hz, loop_freq={loop_freq:.6f} rad/sample ***")

    # -----------------------------------------
    # Run Costas loop
    # -----------------------------------------
    out_I, loop_phase, loop_freq = costas_loop(
        I_raw, Q_raw, loop_phase, loop_freq, alpha_loop, beta_loop
    )

    # Average demodulated output
    avg = np.mean(out_I)
    bit = 1 if avg >= 0 else 0

    freq_hz = loop_freq * FS / (2 * np.pi)
    print(f"BIT = {bit}  avg={avg:.4f}  freq_offset={freq_hz:.0f} Hz  power={power:.4f}")

    display_buffer.append(avg)
    bit_buffer.append(float(bit))

    line1.set_ydata(list(display_buffer))
    line2.set_ydata(list(bit_buffer))
    return line1, line2


ani = FuncAnimation(fig, update, interval=50)
plt.show()
proc.kill()
