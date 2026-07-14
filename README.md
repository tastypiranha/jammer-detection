# Hybrid Jammer Detection Framework

A jammer detection framework for wireless communication systems combining statistical signal processing and polarization-based features. Developed during a Summer Research Fellowship at **IIT Madras** (May–July 2026).

## Overview

The framework addresses jammer detection through three progressively challenging scenarios:

1. **Bayesian Detection** — Known jammer statistics and priors (LRT-based optimal detector)
2. **Minimax Detection** — Unknown prior probabilities (worst-case risk minimization)
3. **Residual-Energy Detection** — Unknown jammer distribution (distribution-free, ~22× energy separation)
4. **Polarization-Based Detection** — Strong jammer regime where signal reconstruction fails

Validated through Monte Carlo simulations, Simulink block-level modeling, and **real-time hardware experiments** using HackRF SDR at 915 MHz.

## Hardware Setup

| Component | Device | Parameters |
|-----------|--------|------------|
| Transmitter | HackRF One | BPSK, 915 MHz, 2 MHz sample rate, 30 dB TX gain |
| Jammer | HackRF One | Sinusoidal/BPSK, 915 MHz |
| Receiver | HackRF One | Python real-time receiver, 20 dB RF gain, 32 dB IF gain |

## Repository Structure

```
├── src/
│   ├── receiver/
│   │   ├── jammer_detector.py      # Full receiver: Costas loop + signal cancellation + jammer extraction
│   │   ├── costas_receiver.py      # BPSK receiver with Costas loop carrier recovery
│   │   ├── fft_receiver.py         # FFT-based BPSK demodulator (differential detection)
│   │   └── jammer_monitor.py       # Raw IQ jammer signal monitor
│   └── transmitter/
│       └── genrator_command.txt     # HackRF TX command reference
├── matlab/
│   ├── signal_generation/
│   │   ├── generator.m             # BPSK IQ file generator for HackRF
│   │   ├── jammer_generator.m      # Jammer signal generator
│   │   └── jamgen.m                # Jammer generation utility
│   └── simulations/
│       ├── intern01.m – intern08.m  # Detection simulations (Bayesian, minimax, residual, polarization)
├── docs/
│   ├── research_paper.tex          # LaTeX source of the research paper
│   ├── research_paper.txt          # Plain-text version of the paper
│   ├── report.tex                  # Project report
│   ├── figures/                    # Simulink block diagrams and parameter plots
│   └── results/                    # Experimental results and simulation outputs
└── README.md
```

## Receiver Architecture

The primary receiver (`src/receiver/jammer_detector.py`) implements:

1. **IQ Acquisition** — Raw 8-bit samples streamed at 2 MHz via `hackrf_transfer`
2. **FFT Frequency Acquisition** — Coarse carrier estimation with parabolic interpolation
3. **Costas Loop** — Second-order PLL (BW = 1 kHz, ζ = 0.707) with Numba JIT for real-time performance
4. **Signal Reconstruction** — Hard-decision baseband re-modulation using tracked phase
5. **Signal Cancellation** — Subtract estimated BPSK from received signal
6. **Jammer Extraction** — 4th-order Butterworth bandpass filter (500 Hz – 100 kHz)
7. **Power Measurement** — Jammer power in dBFS for threshold detection

## Requirements

### Python
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Numba (for JIT-accelerated Costas loop)
- HackRF tools (`hackrf_transfer`)

### MATLAB
- MATLAB R2020a+ with Signal Processing Toolbox
- Simulink (for block-level model)

## Quick Start

### Generate BPSK signal (MATLAB)
```matlab
% Run matlab/signal_generation/generator.m to create bpsk.iq
```

### Transmit
```bash
hackrf_transfer -t bpsk.iq -f 915000000 -s 2000000 -x 30 -a 0
```

### Receive & Detect Jammer
```bash
python src/receiver/jammer_detector.py
```

## Key Results

- **Residual-energy detector**: ~22× energy separation between H0 and H1
- **Polarization detector**: P_D = 0.63 at JSR = 10 dB (complementary to residual method)
- **Hardware validation**: Clear 1 kHz jammer extraction from residual after Costas loop lock

## Detection Approach Comparison

| Method | Needs Jammer Stats? | Works at High JSR? | Hardware Feasible? |
|--------|---------------------|--------------------|--------------------|
| Bayesian/Minimax | Yes | Medium | Yes |
| Residual-energy | No | Weak jammer | Yes |
| Polarization | No | Strong jammer | Harder |

## Author

**Shreyash Puri**  
B.Tech ECE, IIIT Allahabad  
Summer Research Fellow, IIT Madras (2026)

## License

This project is for academic and research purposes.
