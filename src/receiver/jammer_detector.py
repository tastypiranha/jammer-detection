import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from numba import njit
from scipy.signal import butter, lfilter


# ======================
# Numba-accelerated Costas Loop
# ======================

@njit(cache=True)
def costas_loop(I_raw, Q_raw, loop_phase, loop_freq, alpha, beta):
    """
    BPSK Costas loop - returns demodulated I, Q, and loop state.
    Also returns the loop phase at each sample for reconstruction.
    """
    N = len(I_raw)
    out_I = np.empty(N)
    out_Q = np.empty(N)
    phases = np.empty(N)

    for i in range(N):
        cos_v = np.cos(loop_phase)
        sin_v = np.sin(loop_phase)

        i_out = I_raw[i] * cos_v + Q_raw[i] * sin_v
        q_out = -I_raw[i] * sin_v + Q_raw[i] * cos_v

        out_I[i] = i_out
        out_Q[i] = q_out
        phases[i] = loop_phase

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

    return out_I, out_Q, phases, loop_phase, loop_freq


# ======================
# HackRF settings
# ======================

FREQ = "915000000"
FS = 2_000_000

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

LOOP_BW = 1000
zeta = 0.707
BnT = LOOP_BW / FS
denom = 1 + 2*zeta*BnT + BnT**2
alpha_loop = (4 * zeta * BnT) / denom
beta_loop = (4 * BnT**2) / denom

amp_scale = 2.0
alpha_loop *= amp_scale
beta_loop *= amp_scale

print(f"Loop params: alpha={alpha_loop:.6f}, beta={beta_loop:.9f}, BW={LOOP_BW} Hz")


# ======================
# Bandpass filter for jammer extraction
# After subtracting the estimated BPSK, the residual contains:
#   - jammer (narrowband tone)
#   - noise (broadband)
#   - reconstruction error (broadband)
# A bandpass filter isolates the jammer tone.
# ======================

# Bandpass around expected jammer region (0.5 kHz to 100 kHz)
# This passes the jammer while rejecting broadband noise
bp_low = 500.0 / (FS / 2)      # 500 Hz
bp_high = 100000.0 / (FS / 2)  # 100 kHz
b_bp, a_bp = butter(4, [bp_low, bp_high], btype='band')


# ======================
# Parameters
# ======================

CHUNK = 500_000  # 0.25 sec at 2 MHz
N_DISPLAY = 50000  # samples to display on waveform

DISPLAY_SECONDS = 30
CHUNK_DURATION = CHUNK / FS
points_in_window = int(DISPLAY_SECONDS / CHUNK_DURATION)
time_axis_scroll = np.linspace(0, DISPLAY_SECONDS, points_in_window)
time_axis_wave = np.arange(N_DISPLAY) / FS * 1000  # ms


# ======================
# Plot
# ======================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Top: Demodulated BPSK (to confirm signal is being received)
display_buffer = deque([0.0] * points_in_window, maxlen=points_in_window)
line1, = ax1.plot(time_axis_scroll, list(display_buffer), 'b-o', linewidth=2, markersize=3)
ax1.set_title("Demodulated BPSK")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (V, relative)")
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlim(0, DISPLAY_SECONDS)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.grid()

# Middle: Extracted jammer waveform (time domain)
line2, = ax2.plot(time_axis_wave, np.zeros(N_DISPLAY), 'g', linewidth=0.5)
ax2.set_title("Extracted Jammer Signal (after cancellation + filtering)")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Amplitude (V, relative)")
ax2.set_ylim(-0.3, 0.3)
ax2.grid()

# Bottom: Jammer power over time
jammer_power_buf = deque([0.0] * points_in_window, maxlen=points_in_window)
line3, = ax3.plot(time_axis_scroll, list(jammer_power_buf), 'r-o', linewidth=2, markersize=3)
ax3.set_title("Jammer Power")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Power (dBFS)")
ax3.set_ylim(-60, 0)
ax3.set_xlim(0, DISPLAY_SECONDS)
ax3.grid()

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
        return line1, line2, line3

    data = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0
    I_raw = data[0::2]
    Q_raw = data[1::2]
    N = len(I_raw)

    # Check signal power
    power = np.mean(I_raw**2 + Q_raw**2)

    if power < 0.01:
        print(f"NO SIGNAL (power={power:.4f})")
        display_buffer.append(0.0)
        jammer_power_buf.append(0.0)
        line1.set_ydata(list(display_buffer))
        line3.set_ydata(list(jammer_power_buf))
        return line1, line2, line3

    # -----------------------------------------
    # Step 1: Acquire frequency (FFT, once)
    # -----------------------------------------
    if not freq_acquired:
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
        print(f"*** FREQ ACQUIRED: {freq_hz_est:.0f} Hz ***")

    # -----------------------------------------
    # Step 2: Costas loop — demodulate BPSK
    # Returns demod I, Q, and the phase at each sample
    # -----------------------------------------
    out_I, out_Q, phases, loop_phase, loop_freq = costas_loop(
        I_raw, Q_raw, loop_phase, loop_freq, alpha_loop, beta_loop
    )

    # Average demod for bit decision display
    avg = np.mean(out_I)
    bit = 1 if avg >= 0 else 0
    display_buffer.append(avg)

    # -----------------------------------------
    # Step 3: Reconstruct the estimated BPSK signal
    #
    # The demodulated I gives us the baseband BPSK (+/- amplitude).
    # Hard-decision: map to +A or -A where A = measured amplitude.
    # Then re-modulate back to passband using the tracked phase.
    # -----------------------------------------

    # Hard decision on demodulated signal (per-sample)
    amplitude_est = np.mean(np.abs(out_I))  # average signal amplitude
    bpsk_baseband = np.where(out_I > 0, amplitude_est, -amplitude_est)

    # Re-modulate: multiply by carrier (reverse the Costas downconversion)
    # The Costas loop downconverted by multiplying with cos(phase), -sin(phase)
    # To reconstruct: I_est = bpsk * cos(phase), Q_est = -bpsk * sin(phase)
    cos_phases = np.cos(phases)
    sin_phases = np.sin(phases)

    I_estimated = bpsk_baseband * cos_phases
    Q_estimated = -bpsk_baseband * sin_phases

    # -----------------------------------------
    # Step 4: Subtract estimated signal from received
    # residual = received - estimated = jammer + noise
    # -----------------------------------------
    I_residual = I_raw - I_estimated
    Q_residual = Q_raw - Q_estimated

    # -----------------------------------------
    # Step 5: Bandpass filter to extract jammer
    # Removes broadband noise and reconstruction artifacts
    # -----------------------------------------
    I_jammer = lfilter(b_bp, a_bp, I_residual)
    Q_jammer = lfilter(b_bp, a_bp, Q_residual)

    # Jammer power in dBFS
    jammer_pwr = np.mean(I_jammer**2 + Q_jammer**2)
    jammer_pwr_dbfs = 10 * np.log10(jammer_pwr + 1e-10)
    jammer_power_buf.append(jammer_pwr_dbfs)

    freq_hz = loop_freq * FS / (2 * np.pi)
    print(f"BIT = {bit}  avg={avg:.4f}  freq={freq_hz:.0f} Hz  jammer_pwr={jammer_pwr_dbfs:.1f} dBFS")

    # Update plots
    line1.set_ydata(list(display_buffer))

    # Show last N_DISPLAY samples of jammer waveform
    line2.set_ydata(I_jammer[-N_DISPLAY:])

    line3.set_ydata(list(jammer_power_buf))

    return line1, line2, line3


ani = FuncAnimation(fig, update, interval=50)
plt.show()
proc.kill()
