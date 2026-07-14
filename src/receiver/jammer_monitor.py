import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
# Parameters
# ======================

N = 50000  # samples per frame
time_axis = np.arange(N) / FS * 1000  # ms


# ======================
# Plot
# ======================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

line_i, = ax1.plot(time_axis, np.zeros(N), 'b', linewidth=0.5)
ax1.set_title("Jammer - Raw I Channel")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("I")
ax1.set_ylim(-1, 1)
ax1.grid()

line_q, = ax2.plot(time_axis, np.zeros(N), 'r', linewidth=0.5)
ax2.set_title("Jammer - Raw Q Channel")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Q")
ax2.set_ylim(-1, 1)
ax2.grid()

plt.tight_layout()


# ======================
# Update
# ======================

def update(frame):
    raw = proc.stdout.read(2 * N)
    if len(raw) < 2 * N:
        return line_i, line_q

    data = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0
    I = data[0::2]
    Q = data[1::2]

    line_i.set_ydata(I)
    line_q.set_ydata(Q)
    return line_i, line_q


ani = FuncAnimation(fig, update, interval=30)
plt.show()
proc.kill()
