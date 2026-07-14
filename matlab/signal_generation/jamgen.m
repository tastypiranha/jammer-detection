%% Generate BPSK Jammer for HackRF
%% Low power BPSK signal acting as a jammer
clear; clc;

Fs = 2e6;
bitDuration = 5;      % each bit 5 seconds
totalTime = 30;

% Jammer power: 30% of main signal
jammerPower = 0.30;
jammerAmp = round(127 * sqrt(jammerPower));

% Random bit pattern (different from main signal)
bits = [0 1 1 0 1 0];

samplesPerBit = Fs * bitDuration;

fid = fopen('jammer.iq', 'wb');
fprintf("Generating BPSK jammer...\n")
fprintf("Amplitude: %d (out of 127)\n", jammerAmp)
fprintf("Power ratio: %.0f%%\n", jammerPower*100)

for k = 1:length(bits)
    if bits(k) == 1
        symbol = jammerAmp;
    else
        symbol = -jammerAmp;
    end

    I = int8(symbol * ones(samplesPerBit, 1));
    Q = int8(zeros(samplesPerBit, 1));

    iq = zeros(2*samplesPerBit, 1, 'int8');
    iq(1:2:end) = I;
    iq(2:2:end) = Q;

    fwrite(fid, iq, 'int8');
    fprintf("Bit %d = %d\n", k, bits(k))
end

fclose(fid);
fprintf("Done!\n")
fprintf("hackrf_transfer -t jammer.iq -f 915000000 -s 2000000 -x 30 -a 0\n")
